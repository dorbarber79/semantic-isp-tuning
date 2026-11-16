"""Download/fit an authorized FiveK pilot and queue interactive visual annotations."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading
import uuid

from fivek_batch import download_image, save_json, sha256, validate_result, write_index
from fivek_photographer_fit import DEFAULT_CONFIG, DEFAULT_DATA, DEFAULT_FIVEK, PHOTOGRAPHERS


DEFAULT_OUTPUT = DEFAULT_DATA / 'results/isp_match_20_pilot_v1'
DEFAULT_PREVIOUS = DEFAULT_DATA / 'results/isp_match_105_v4/index.json'
RUNNER = Path(__file__).with_name('fivek_photographer_fit.py')


def pilot_selection(data_root, fivek_root, previous_index, count, seed):
    licensed = set((data_root / 'filesAdobeMIT.txt').read_text().splitlines())
    excluded = {entry['image_id'] for entry in json.loads(previous_index.read_text())['images']}
    available = sorted(name for name in licensed - excluded if name and '/' not in name and '\\' not in name
                       and (fivek_root / 'default_isp_linear_720p' / f'{name}.png').is_file())
    if count < 1 or len(available) < count:
        raise ValueError('Insufficient licensed new scenes with existing Default ISP renders')
    return random.Random(seed).sample(available, count)


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    image_ids = pilot_selection(args.data_root, args.fivek_root, args.previous, args.count, args.seed)
    manifest = {'pilot_images': image_ids, 'authorized_maximum': args.count,
                'seed': args.seed, 'previous_index': str(args.previous.resolve()),
                'previous_index_sha256': sha256(args.previous),
                'license': str(args.data_root / 'LicenseAdobeMIT.txt'),
                'license_list_sha256': sha256(args.data_root / 'filesAdobeMIT.txt'),
                'config': str(args.config.resolve()), 'config_sha256': sha256(args.config),
                'runner_sha256': sha256(RUNNER), 'worker_sha256': sha256(Path(__file__)),
                'iterations': args.iterations, 'learning_rate': 0.03,
                'device': args.device, 'annotation_mode': 'Interactive visual-review agent only; no automatic VLM labels'}
    manifest_path = args.out / 'collection_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Collection source/config changed; use a new output directory')
    else:
        if any(path.name != '.worker.lock' for path in args.out.iterdir()):
            raise FileExistsError('Refusing an existing directory without a collection manifest')
        save_json(manifest_path, manifest)
    (args.out / 'images').mkdir(exist_ok=True)
    state_path = args.out / 'collection_state.json'
    if not state_path.exists():
        save_json(state_path, {'worker_state': 'prepared', 'authorized_images': args.count,
                              'jobs': {image_id: {'fit_state': 'pending', 'annotation_state': 'waiting_for_fit'}
                                       for image_id in image_ids}})
    annotation_path = args.out / 'fit_visual_descriptions.json'
    queue_path = args.out / 'annotation_queue.json'
    if not queue_path.exists():
        save_json(queue_path, {'annotation_mode': 'Interactive review required', 'scenes': []})
    if not annotation_path.exists():
        save_json(annotation_path, {
            'schema_version': 1,
            'method': 'Assistant-authored interactive visual review of fitted ISP output relative to Default ISP.',
            'limitations': 'Pending fits have no generated descriptions; existing 105-scene labels are unchanged.',
            'annotation_process': {'mode': 'Interactive review only', 'source_fit_run': str(args.out),
                                   'model_identifier': 'Record actual available annotator metadata per scene; do not infer it.',
                                   'scene_provenance': {}}, 'images': {}})
    return manifest


def annotation_status(image_id, document):
    fits = document['images'].get(image_id)
    if fits is None:
        return 'pending_visual_review'
    if set(fits) != set(PHOTOGRAPHERS):
        return 'incomplete_annotation'
    for expert in PHOTOGRAPHERS:
        if set(fits[expert]) != {'D1', 'D2', 'D3', 'D4'}:
            return 'incomplete_annotation'
        for group, text in fits[expert].items():
            if not isinstance(text, str) or not text.strip():
                return 'incomplete_annotation'
            if group == 'D1' and len(text.split()) > 10:
                return 'invalid_annotation'
            if group == 'D3' and len(text.split()) > 11:
                return 'invalid_annotation'
    for group in ('D1', 'D2', 'D3', 'D4'):
        if len({fits[expert][group].strip().casefold() for expert in PHOTOGRAPHERS}) != 5:
            return 'invalid_annotation'
    provenance = document.get('annotation_process', {}).get('scene_provenance', {}).get(image_id, {})
    required = ('annotator_surface', 'model_identifier', 'visual_inputs', 'guidance', 'source_result_sha256')
    if not all(isinstance(provenance.get(key), str) and provenance[key].strip() for key in required):
        return 'missing_provenance'
    return 'reviewed'


def update_review_queue(output, state):
    document = json.loads((output / 'fit_visual_descriptions.json').read_text())
    queue = []
    for image_id, job in state['jobs'].items():
        if job['fit_state'] != 'complete':
            continue
        folder = output / 'images' / image_id
        status = annotation_status(image_id, document)
        if status == 'reviewed':
            provenance = document['annotation_process']['scene_provenance'][image_id]
            if provenance['source_result_sha256'] != sha256(folder / 'results.json'):
                status = 'source_changed'
        job['annotation_state'] = status
        result = json.loads((folder / 'results.json').read_text())
        fits = result['per_image'][image_id]
        queue.append({'image_id': image_id, 'status': status,
                      'results': str(folder / 'results.json'),
                      'source_result_sha256': sha256(folder / 'results.json'),
                      'comparison_sheet': str(folder / 'figures' / f'{image_id}_isp_fit_comparison.png'),
                      'default_render': fits['A']['neutral_display_path'],
                      'fit_renders': {expert: fits[expert]['fit_display_path'] for expert in PHOTOGRAPHERS},
                      'annotation_file': str(output / 'fit_visual_descriptions.json')})
    save_json(output / 'annotation_queue.json', {'annotation_mode': 'Interactive review required', 'scenes': queue})
    return queue


def work(args):
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another collection worker already owns this directory') from error
        manifest = prepare_locked(args)
        state_path = args.out / 'collection_state.json'
        state = json.loads(state_path.read_text())
        for job in state['jobs'].values():
            if job['fit_state'] in ('downloading', 'fitting'):
                job['fit_state'] = 'pending'
        if args.retry_failed:
            for job in state['jobs'].values():
                if job['fit_state'] == 'failed':
                    job['fit_state'] = 'pending'
        stop = threading.Event()
        child = None

        def request_stop(signum, frame):
            stop.set()
            if child is not None and child.poll() is None:
                child.terminate()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        save_json(args.out / 'worker_pid.json', {'pid': os.getpid(), 'output': str(args.out)})
        last_queue_hash = None
        try:
            while not stop.is_set() and not (args.out / 'STOP').exists():
                pending = next((image_id for image_id, job in state['jobs'].items()
                                if job['fit_state'] == 'pending'), None)
                if pending:
                    job = state['jobs'][pending]
                    folder = args.out / 'images' / pending
                    state['worker_state'] = 'running'
                    try:
                        if folder.exists() and validate_result(folder, pending) is not None:
                            job['fit_state'] = 'complete'
                        else:
                            if shutil.disk_usage(args.out).free < args.min_free_gib * 1024 ** 3:
                                raise RuntimeError('Free disk space is below the configured reserve')
                            if folder.exists():
                                folder.rename(folder.with_name(pending + '.incomplete-' + uuid.uuid4().hex))
                            job['fit_state'] = 'downloading'
                            save_json(state_path, state)
                            print(f'Downloading {pending}', flush=True)
                            download_image(args.data_root, pending)
                            if stop.is_set():
                                job['fit_state'] = 'pending'
                                break
                            job['fit_state'] = 'fitting'
                            save_json(state_path, state)
                            command = [sys.executable, str(RUNNER), '--data-root', str(args.data_root),
                                       '--fivek-root', str(args.fivek_root), '--config', str(args.config),
                                       '--out', str(folder), '--image-ids', pending,
                                       '--photographers', *PHOTOGRAPHERS, '--iterations', str(args.iterations),
                                       '--learning-rate', '0.03', '--device', args.device]
                            child = subprocess.Popen(command)
                            return_code = child.wait()
                            child = None
                            if stop.is_set():
                                job['fit_state'] = 'pending'
                                break
                            if return_code:
                                raise RuntimeError(f'Fit command failed with exit code {return_code}')
                            validate_result(folder, pending)
                            job['fit_state'] = 'complete'
                        print(f'Completed fit {pending}', flush=True)
                    except Exception as error:
                        job['fit_state'], job['error'] = 'failed', str(error)
                        print(f'Failed {pending}: {error}', flush=True)
                    write_index(args.out, manifest['pilot_images'])
                try:
                    queue = update_review_queue(args.out, state)
                    state.pop('annotation_warning', None)
                except (json.JSONDecodeError, OSError) as error:
                    queue = []
                    state['annotation_warning'] = str(error)
                queue_hash = sha256(args.out / 'annotation_queue.json') if queue else None
                valid_queue = all(entry['status'] in ('reviewed', 'pending_visual_review') for entry in queue)
                if queue and valid_queue and queue_hash != last_queue_hash:
                    from fivek_screen_comparisons import process
                    process(args.out, args.out / 'screen_report', args.out / 'fit_visual_descriptions.json')
                    last_queue_hash = queue_hash
                state['completed_fits'] = sum(job['fit_state'] == 'complete' for job in state['jobs'].values())
                state['reviewed_scenes'] = sum(job.get('annotation_state') == 'reviewed' for job in state['jobs'].values())
                if not pending:
                    state['worker_state'] = 'idle_waiting_for_visual_review'
                save_json(state_path, state)
                if not pending:
                    if not args.watch:
                        break
                    stop.wait(args.interval)
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            state['worker_state'] = 'stopped'
            save_json(state_path, state)


def prepare_locked(args):
    manifest_path = args.out / 'collection_manifest.json'
    if not manifest_path.exists():
        lock = args.out / '.worker.lock'
        if any(path != lock for path in args.out.iterdir()):
            raise FileExistsError('Collection directory already contains unrelated files')
    return prepare(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--fivek-root', type=Path, default=DEFAULT_FIVEK)
    parser.add_argument('--previous', type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--min-free-gib', type=float, default=20)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.count < 1 or args.iterations < 1 or args.interval <= 0 or args.min_free_gib <= 0:
        parser.error('Counts, iterations, interval and disk reserve must be positive')
    if args.prepare_only:
        manifest = prepare(args)
        print(json.dumps(manifest, indent=2))
    else:
        work(args)


if __name__ == '__main__':
    main()