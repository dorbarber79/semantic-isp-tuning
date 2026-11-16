import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from autotune import CONTROL_NAMES, objective_terms, optimize, paired_quality_scores
from benchmark import load_dataset
from exp49_direct_isp_comparison import direct_config
from generate_dataset import run as generate_dataset, validate_sentences
from fivek_preview import load_linear_png, resize_short_side
from fivek_semantic_descriptions import AXES, DEFAULT_WORDS, descriptions, read_word_bank, style_candidates
from fivek_screen_comparisons import fit_descriptor_record, load_fit_visual_descriptions, screen_fit_record
from fivek_visual_report import ANNOTATIONS, FILTER_NAMES
from semantic_isp import CONTROL_BOUNDS, neutral_raw_controls
from pole_normalization import normalized_to_physical, physical_to_normalized, pole_endpoints
from fivek_clip_svr import comparison_rows, descriptor_inputs, fused_features, make_regressor, scene_folds, select_experts
from fivek_mlp import choose_experts as choose_mlp_experts, fusion_features as mlp_fusion_features
from fivek_brightness_svr import brightness_controls, isolated_controls, trend_metrics
from fivek_text_linear_report import prediction_scores
from fivek_all_descriptors_svr import all_descriptor_text, comparison_metrics
from fivek_collection_worker import annotation_status, pilot_selection
from package_fivek_dataset import ReleaseStore, annotation_layers, build_unified_gallery, control_statistics, file_hash, merge_fit_descriptions
from fivek_dataset_psnr_statistics import trim_lowest


class ToyModel(torch.nn.Module):
    def encode_image(self, image):
        means = image.mean(dim=(-1, -2))
        return torch.cat((means, means.std(dim=1, keepdim=True)), dim=1)


class ExperimentTests(unittest.TestCase):
    def test_psnr_trim_removes_exact_lowest_fraction(self):
        rows = [{'image_id': f'scene_{index}', 'expert': 'A', 'psnr_db': float(index)}
                for index in range(20)]
        retained, removed = trim_lowest(rows, .05)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]['psnr_db'], 0.0)
        self.assertEqual(len(retained), 19)
        retained, removed = trim_lowest(rows, .10)
        self.assertEqual({row['psnr_db'] for row in removed}, {0.0, 1.0})

    def test_mlp_features_and_original_expert_selection(self):
        image = np.full((2, 512), 1 / np.sqrt(512))
        text = np.full((2, 512), -1 / np.sqrt(512))
        self.assertEqual(mlp_fusion_features(image, text).shape, (2, 2048))
        original = [{'image_id': f'old_{index}', 'expert': 'ABCDE'[index % 5]}
                    for index in range(105)]
        all_scenes = [row['image_id'] for row in original] + [f'new_{index}' for index in range(20)]
        selected = choose_mlp_experts(all_scenes, original, seed=17)
        self.assertTrue(all(selected[row['image_id']] == row['expert'] for row in original))
        counts = {expert: sum(selected[f'new_{index}'] == expert for index in range(20))
                  for expert in 'ABCDE'}
        self.assertEqual(set(counts.values()), {4})

    def test_unified_gallery_uses_packaged_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fits = {expert: {group: f'{expert} {group} note.' for group in ('D1', 'D2', 'D3', 'D4')}
                    for expert in 'ABCDE'}
            rows = [{'image_id': 'scene', 'expert': expert, 'annotation_layers': [],
                     'fit_metadata': {'normalized_controls': dict.fromkeys(CONTROL_NAMES, 0)},
                     'assets': {role: {'path': f'scenes/scene/{role}.png'}
                                for role in ('default_display_720', 'expert_display_720', 'fit_display_720')}}
                    for expert in 'ABCDE']
            with patch('fivek_screen_comparisons.draw_comparison') as draw:
                build_unified_gallery(root, rows, {'images': {'scene': fits}})
            figure_rows = draw.call_args.args[2]
            self.assertEqual(len(figure_rows), 5)
            self.assertEqual(figure_rows[0]['fit']['fit_display_path'],
                             str(root / 'scenes/scene/fit_display_720.png'))
            self.assertTrue((root / 'screen_report/index.html').exists())

    def test_merged_fit_descriptions_keep_source_provenance(self):
        fits = {expert: {group: f'{expert} {group} exact phrase.'
                         for group in ('D1', 'D2', 'D3', 'D4')} for expert in 'ABCDE'}
        documents = [('original.json', {'images': {'old': fits},
                     'annotation_process': {'underlying_model_identifier': 'Unknown'}}),
                     ('pilot.json', {'images': {'new': fits},
                     'annotation_process': {'scene_provenance': {'new': {'model_identifier': 'Unavailable'}}}})]
        merged = merge_fit_descriptions(documents)
        self.assertEqual(set(merged['images']), {'old', 'new'})
        self.assertEqual(merged['images']['old'], fits)
        self.assertEqual(merged['annotation_process']['snapshot_counts']['reviewed_fits'], 10)
        self.assertEqual(merged['annotation_process']['scene_provenance']['new']
                         ['original_scene_provenance']['model_identifier'], 'Unavailable')
        with self.assertRaises(ValueError):
            merge_fit_descriptions([documents[0], documents[0]])

    def test_dataset_release_copies_source_and_deduplicates_internally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.json'
            source.write_text(json.dumps({'text': 'Preserve this exact phrase.'}))
            store = ReleaseStore(root / 'release')
            first = store.add(source, 'annotations/a.json')
            second = store.add(source, 'archives/b.json')
            self.assertEqual(file_hash(source), file_hash(root / 'release' / first))
            self.assertNotEqual(source.stat().st_ino, (root / 'release' / first).stat().st_ino)
            self.assertEqual((root / 'release' / first).stat().st_ino, (root / 'release' / second).stat().st_ino)
            source.unlink()
            self.assertTrue((root / 'release' / first).exists())
            with self.assertRaises(ValueError):
                store.add(root / 'release' / first, '../outside.json')

    def test_dataset_release_relinks_existing_duplicate_to_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.html'
            source.write_text('<h1>Report</h1>')
            release = root / 'release'
            destination = release / 'reports/latest/index.html'
            destination.parent.mkdir(parents=True)
            destination.write_text(source.read_text())
            store = ReleaseStore(release)
            relative = store.add(source, 'reports/latest/index.html')
            object_path = release / 'objects' / store.assets[relative]['sha256'][:2] / store.assets[relative]['sha256']
            self.assertEqual(destination.stat().st_ino, object_path.stat().st_ino)

    def test_dataset_annotation_layers_preserve_distinct_targets(self):
        document = {'images': {'scene': {'A': {'D1': 'Fit phrase.'}}},
                    'annotations': {'scene': {'A': {'observation': 'Expert observation.'}}},
                    'annotation_process': {'model_identifier': 'Unavailable'}}
        layers = annotation_layers('scene', 'A', [('original.json', document)])
        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0]['value']['D1'], 'Fit phrase.')
        self.assertEqual(layers[1]['value']['observation'], 'Expert observation.')
        self.assertNotEqual(layers[0]['annotation_target'], layers[1]['annotation_target'])
        self.assertEqual(layers[0]['global_annotation_process']['model_identifier'], 'Unavailable')

    def test_dataset_control_statistics_separate_domains_and_experts(self):
        rows = [{'expert': 'A', 'normalized_controls': dict.fromkeys(CONTROL_NAMES, 0.2),
                 'physical_controls': dict.fromkeys(CONTROL_NAMES, 0.4)},
                {'expert': 'B', 'normalized_controls': dict.fromkeys(CONTROL_NAMES, -0.3),
                 'physical_controls': dict.fromkeys(CONTROL_NAMES, 0.6)}]
        summary = control_statistics(rows)
        self.assertAlmostEqual(summary['all']['normalized_controls']['warmth']['mean'], -0.05)
        self.assertAlmostEqual(summary['all']['physical_controls']['warmth']['mean'], 0.5)
        self.assertEqual(summary['A']['normalized_controls']['warmth']['count'], 1)
        self.assertAlmostEqual(summary['B']['normalized_controls']['warmth']['mean'], -0.3)

    def test_collection_annotation_requires_visual_provenance(self):
        document = {'images': {}}
        self.assertEqual(annotation_status('scene', document), 'pending_visual_review')
        document['images']['scene'] = {expert: {group: f'{expert} {group} visual note.'
                    for group in ('D1', 'D2', 'D3', 'D4')} for expert in 'ABCDE'}
        self.assertEqual(annotation_status('scene', document), 'missing_provenance')
        document['annotation_process'] = {'scene_provenance': {'scene': {
            key: 'recorded' for key in ('annotator_surface', 'model_identifier', 'visual_inputs',
                                       'guidance', 'source_result_sha256')}}}
        self.assertEqual(annotation_status('scene', document), 'reviewed')
        document['images']['scene']['B']['D1'] = document['images']['scene']['A']['D1']
        self.assertEqual(annotation_status('scene', document), 'invalid_annotation')

    def test_collection_pilot_excludes_previous_and_unavailable_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'filesAdobeMIT.txt').write_text('previous\nnew_a\nnew_b\nmissing\n../unsafe\n')
            (root / 'default_isp_linear_720p').mkdir()
            for image_id in ('previous', 'new_a', 'new_b'):
                (root / 'default_isp_linear_720p' / f'{image_id}.png').touch()
            previous = root / 'previous.json'
            previous.write_text(json.dumps({'images': [{'image_id': 'previous'}]}))
            selected = pilot_selection(root, root, previous, 2, 0)
            self.assertEqual(set(selected), {'new_a', 'new_b'})
            with self.assertRaises(ValueError):
                pilot_selection(root, root, previous, 3, 0)

    def test_pilot_annotation_loader_accepts_separate_empty_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pilot.json'
            path.write_text(json.dumps({'images': {}}))
            self.assertEqual(load_fit_visual_descriptions(path)['images'], {})

    def test_all_descriptor_comparison_uses_exact_groups(self):
        groups = {'D1': 'one', 'D2': 'two', 'D3': 'three', 'D4': 'four'}
        self.assertEqual(all_descriptor_text(groups), 'one two three four')
        target = np.asarray([-0.5, 0, 0.5])
        metrics = comparison_metrics(target, np.zeros(3), target, np.zeros(3))
        self.assertAlmostEqual(metrics['all_d_relative_mae_improvement_percent'], 100)
        self.assertEqual(metrics['D1_D4']['within_0_1_percent'], 100)
        with self.assertRaises(ValueError):
            all_descriptor_text({'D1': 'only one'})

    def test_composed_report_parameter_and_image_scores(self):
        target = np.zeros((2, 2, 3), dtype=np.uint8)
        predicted = np.full((2, 2, 3), 10, dtype=np.uint8)
        scores = prediction_scores(np.zeros(6), np.full(6, 0.2), target, predicted)
        self.assertAlmostEqual(scores['parameter_mae'], 0.2)
        self.assertEqual(set(scores['parameter_absolute_errors']), set(CONTROL_NAMES))
        self.assertTrue(all(np.isclose(value, 0.2) for value in scores['parameter_absolute_errors'].values()))
        self.assertEqual(scores['image_mae_255'], 10)
        self.assertAlmostEqual(scores['image_psnr_db'], 20 * np.log10(255 / 10))
        self.assertIsNone(prediction_scores(np.zeros(6), np.zeros(6), target, target)['image_psnr_db'])

    def test_composed_report_shows_expert_fit_and_prediction(self):
        from fivek_screen_comparisons import draw_comparison
        controls = dict.fromkeys(CONTROL_NAMES, 0.0)
        row = {'photographer': 'A', 'input_text': 'Warm light.',
               'prediction_display_path': 'predicted.png', 'target_controls': controls,
               'descriptors': {'fit_based': True},
               'scores': {'parameter_mae': 0.1, 'image_mae_255': 10, 'image_psnr_db': 20},
               'fit': {'neutral_display_path': 'default.png', 'display_path': 'expert.png',
                       'fit_display_path': 'fit.png', 'normalized_controls': controls}}
        with tempfile.TemporaryDirectory() as directory:
            with patch('fivek_screen_comparisons.paste_image') as paste:
                draw_comparison(Path(directory) / 'comparison.png', 'scene', [row])
            self.assertEqual([call.args[1] for call in paste.call_args_list],
                             ['default.png', 'expert.png', 'fit.png', 'predicted.png'])

    def test_brightness_study_isolates_one_control(self):
        controls = brightness_controls(-0.4)
        self.assertEqual(controls['tone_gain'], -0.4)
        self.assertTrue(all(value == 0 for name, value in controls.items() if name != 'tone_gain'))
        for control in CONTROL_NAMES:
            isolated = isolated_controls(control, 0.3)
            self.assertEqual(isolated[control], 0.3)
            self.assertTrue(all(value == 0 for name, value in isolated.items() if name != control))
        with self.assertRaises(ValueError):
            isolated_controls('unknown', 0.3)
        target = np.asarray([-0.4, 0.2, 0.8])
        self.assertEqual(trend_metrics(target, target)['mae'], 0)
        self.assertAlmostEqual(trend_metrics(target, target)['r2'], 1)
        self.assertIsNone(trend_metrics(target, np.zeros(3))['pearson_r'])

    def test_svr_comparison_uses_fit_target_and_exact_input(self):
        target = {name: 0.1 for name in CONTROL_NAMES}
        predicted = {name: 0.2 for name in CONTROL_NAMES}
        result = {'expert': 'A', 'reference_fit': {'display_path': 'expert.png',
                  'fit_display_path': 'target_fit.png', 'normalized_controls': target},
                  'predictions': [{'variant': 'D1', 'text': 'The exact input phrase.',
                                   'render': 'prediction.png', 'normalized_controls': predicted}]}
        row = comparison_rows(result)[0]
        self.assertEqual(row['fit']['display_path'], 'target_fit.png')
        self.assertEqual(row['fit']['fit_display_path'], 'prediction.png')
        self.assertEqual(row['input_text'], 'The exact input phrase.')
        self.assertEqual(row['target_controls'], target)
        self.assertEqual(row['fit']['normalized_controls'], predicted)

    def test_svr_expert_selection_and_scene_folds(self):
        image_ids = [f"scene_{index}" for index in range(10)]
        selected = select_experts(image_ids)
        self.assertEqual(selected, select_experts(list(reversed(image_ids))))
        self.assertTrue(all(list(selected.values()).count(expert) == 2 for expert in "ABCDE"))
        groups = np.repeat(image_ids, 5)
        folds = list(scene_folds(groups))
        for training, testing in folds:
            self.assertFalse(set(groups[training]) & set(groups[testing]))
        self.assertEqual(sorted(np.concatenate([testing for training, testing in folds]).tolist()),
                         list(range(len(groups))))

    def test_svr_features_and_six_control_outputs(self):
        features = fused_features(np.ones((10, 8)), np.full((10, 8), 2.0))
        np.testing.assert_allclose(np.linalg.norm(features[:, :8], axis=1), 1)
        np.testing.assert_allclose(np.linalg.norm(features[:, 8:], axis=1), 1)
        self.assertEqual(make_regressor().fit(features, np.zeros((10, 6))).predict(features).shape, (10, 6))
        with self.assertRaises(ValueError):
            fused_features(np.zeros((10, 8)), np.ones((10, 8)))

    def test_svr_descriptor_inputs_keep_group_text(self):
        groups = {group: f"Example {group} description." for group in ("D1", "D2", "D3", "D4")}
        inputs = descriptor_inputs(groups, "Quiet Shore", [{"app": "Instagram", "name": "Gingham"}])
        for group, text in groups.items():
            self.assertEqual(inputs[group], text)
            self.assertIn(text, inputs["combined"])
        self.assertIn("Quiet Shore", inputs["combined"])
        self.assertIn("Instagram Gingham", inputs["combined"])

    def test_visual_filter_names_cover_annotations(self):
        annotations = json.loads(ANNOTATIONS.read_text())["annotations"]
        filter_data = json.loads(FILTER_NAMES.read_text())
        self.assertIn("not Instagram", filter_data["method"])
        names = filter_data["names"]
        self.assertEqual(set(names), set(annotations))
        for image_id, experts in annotations.items():
            self.assertEqual(set(names[image_id]), set(experts))
            self.assertEqual(len(set(names[image_id].values())), len(experts))
            self.assertTrue(all(name.strip() for name in names[image_id].values()))



    def test_manual_fit_visual_descriptions_load_verbatim(self):
        document = load_fit_visual_descriptions()
        counts = document["annotation_process"]["snapshot_counts"]
        self.assertEqual(len(document["images"]), counts["reviewed_images"])
        self.assertEqual(counts["reviewed_fits"], len(document["images"]) * 5)
        image_id = "a0015-DSC_0081"
        descriptor = fit_descriptor_record(image_id, "A", document, [], "")
        self.assertTrue(descriptor["fit_based"])
        self.assertEqual(descriptor["groups"]["D1"]["variants"][0],
                         "Cooler light keeps the beach subdued.")
        self.assertIn("Pale blue sea", descriptor["groups"]["D3"]["variants"][0])
        self.assertNotIn("expert", " ".join(
            phrase for group in descriptor["groups"].values() for phrase in group["variants"]).lower())
        flag_fit = fit_descriptor_record("a1389-IMG_4831", "D", document, [], "")
        self.assertIn("Ivory", flag_fit["groups"]["D3"]["variants"][0])
        plane_fit = fit_descriptor_record("a4002-IMG_4605", "E", document, [], "")
        self.assertIn("Airplane nearly fades", plane_fit["groups"]["D3"]["variants"][0])
        sunset_fit = fit_descriptor_record("a4540-Duggan_080411_5948", "E", document, [], "")
        self.assertIn("horizon glows", sunset_fit["groups"]["D3"]["variants"][0])
        portrait_fit = fit_descriptor_record("a0731-IMG_3972", "A", document, [], "")
        self.assertIn("green bokeh", portrait_fit["groups"]["D3"]["variants"][0])
        gallery_fit = fit_descriptor_record("a4792-20090417_at_18h06m12__MG_1956", "A", document, [], "")
        self.assertIn("ribbed sculpture", gallery_fit["groups"]["D3"]["variants"][0])
        bridge_fit = fit_descriptor_record("a1409-IMG_4840", "C", document, [], "")
        self.assertIn("cyan sky", bridge_fit["groups"]["D1"]["variants"][0])
        marine_fit = fit_descriptor_record("a1032-IMG_1542", "B", document, [], "")
        self.assertIn("turquoise waves", marine_fit["groups"]["D3"]["variants"][0])
        meeting_fit = fit_descriptor_record("a0568-_MG_1090", "E", document, [], "")
        self.assertIn("bright monochrome", meeting_fit["groups"]["D3"]["variants"][0])
        fruit_fit = fit_descriptor_record("a4237-20090417_at_18h06m01__MG_1955", "A", document, [], "")
        self.assertIn("plum", fruit_fit["groups"]["D3"]["variants"][0])
        camera_fit = fit_descriptor_record("a0606-051002_152054__I2E8779", "D", document, [], "")
        self.assertIn("camera", camera_fit["groups"]["D3"]["variants"][0])
        event_fit = fit_descriptor_record("a4955-_DGW0261", "B", document, [], "")
        self.assertIn("pale shirts", event_fit["groups"]["D3"]["variants"][0])
        store_fit = fit_descriptor_record("a4724-Duggan_090319_5593", "A", document, [], "")
        self.assertIn("Palm Beach storefront", store_fit["groups"]["D3"]["variants"][0])
        installation_fit = fit_descriptor_record("a3303-_DSC0060", "A", document, [], "")
        self.assertIn("folding chair", installation_fit["groups"]["D3"]["variants"][0])
        yacht_fit = fit_descriptor_record("a4745-Duggan_090330_6275", "A", document, [], "")
        self.assertIn("White yacht", yacht_fit["groups"]["D3"]["variants"][0])
        child_fit = fit_descriptor_record("a4437-20090326_at_03h22m38__MG_0355", "A", document, [], "")
        self.assertIn("red sofa", child_fit["groups"]["D3"]["variants"][0])
        sail_fit = fit_descriptor_record("a2863-IMG_0090", "C", document, [], "")
        self.assertIn("white sail", sail_fit["groups"]["D3"]["variants"][0])
        flower_fit = fit_descriptor_record("a4672-kme_0259", "B", document, [], "")
        self.assertIn("White blossoms", flower_fit["groups"]["D3"]["variants"][0])
        forest_fit = fit_descriptor_record("a4249-Duggan_090322_6001", "A", document, [], "")
        self.assertIn("luminous greenery", forest_fit["groups"]["D3"]["variants"][0])
        street_fit = fit_descriptor_record("a3648-07-06-01-at-12h59m03-s_MG_1251", "C", document, [], "")
        self.assertIn("Bicycle and riders", street_fit["groups"]["D3"]["variants"][0])
        sign_fit = fit_descriptor_record("a3077-IMG_4837", "A", document, [], "")
        self.assertIn("400 numerals", sign_fit["groups"]["D3"]["variants"][0])
        granite_fit = fit_descriptor_record("a1401-DSC_0034-1", "C", document, [], "")
        self.assertIn("cobalt sky", granite_fit["groups"]["D3"]["variants"][0])
        gull_fit = fit_descriptor_record("a1173-IMG_1748", "A", document, [], "")
        self.assertIn("terracotta tiles", gull_fit["groups"]["D3"]["variants"][0])
        taxi_fit = fit_descriptor_record("a3312-IMG_1063", "C", document, [], "")
        self.assertIn("dark storefronts", taxi_fit["groups"]["D3"]["variants"][0])
        car_fit = fit_descriptor_record("a3248-DSC_0023-2", "A", document, [], "")
        self.assertIn("Chrome headlight", car_fit["groups"]["D3"]["variants"][0])

    def test_fit_annotation_provenance_records_unknown_model_version(self):
        data_path = Path(__file__).with_name("fivek_fit_visual_descriptions.json")
        process = json.loads(data_path.read_text())["annotation_process"]
        self.assertEqual(process["annotator_surface"], "GitHub Copilot Chat")
        self.assertIn("Not exposed", process["underlying_model_identifier"])
        self.assertIn("GPT-5.6 Luna", process["model_name_reported_by_user"])
        self.assertIn("not independently confirmed", process["model_name_reported_by_user"])
        self.assertIn("No fixed per-image model prompt", process["prompt_record"])
        counts = process["snapshot_counts"]
        self.assertEqual(counts["reviewed_images"], len(json.loads(data_path.read_text())["images"]))
        self.assertEqual(counts["reviewed_fits"], counts["reviewed_images"] * 5)
        self.assertEqual(counts["unreviewed_images"], counts["fit_source_images"] - counts["reviewed_images"])
        self.assertEqual(counts["unreviewed_fits"], counts["unreviewed_images"] * 5)

    def test_manual_fit_descriptions_are_concise(self):
        document = load_fit_visual_descriptions()
        all_lengths = []
        for fits in document["images"].values():
            for groups in fits.values():
                all_lengths.extend(len(text.split()) for text in groups.values())
                self.assertLessEqual(len(groups["D1"].split()), 10)
                self.assertLessEqual(len(groups["D3"].split()), 11)
        self.assertGreaterEqual(sum(length < 12 for length in all_lengths), .9 * len(all_lengths))

    def test_unreviewed_fit_has_pending_notice_not_generated_prose(self):
        document = load_fit_visual_descriptions()
        descriptor = fit_descriptor_record("unreviewed-image", "A", document, [], "")
        self.assertFalse(descriptor["fit_based"])
        self.assertEqual(descriptor["status"], "Visual fit description pending")
        self.assertTrue(all(not descriptor["groups"][group]["variants"]
                            for group in ("D1", "D2", "D3", "D4")))

    def test_screen_fit_record_omits_legacy_descriptors(self):
        fit = {"descriptors": {"D1": "legacy"}, "agent_prompt": "legacy", "normalized_controls": {}}
        result = screen_fit_record(fit)
        self.assertNotIn("descriptors", result)
        self.assertNotIn("agent_prompt", result)
        self.assertIn("normalized_controls", result)

    def test_fivek_human_descriptors_use_semantic_bank(self):
        words = read_word_bank(DEFAULT_WORDS)
        values = {"warmth": .2, "saturation": -.3, "tone_gain": .6,
                  "contrast": -.5, "sharpness": -.5, "palette": 0.}
        result = descriptions(values, words)
        self.assertIn("angelic", [row["word"] for row in result["semantic_candidates"]])
        for group in ("D1", "D2", "D3", "D4"):
            variants = result["groups"][group]["variants"]
            self.assertEqual(len(set(variants)), 4)
            for text in variants:
                self.assertNotRegex(text.lower(), r"\b(?:white balance|tone gain|saturation|sharpness|isp)\b")
        conflicting = {**values, "tone_gain": -.7}
        self.assertNotIn("angelic", [row["word"] for row in descriptions(conflicting, words)["semantic_candidates"]])

    def test_style_candidates_fill_with_nearest_direction_conflicts(self):
        values = {axis: 1.0 for axis in AXES}
        words = [{"word": word, "ratings": {axis: -1.0 for axis in AXES}, "appearance": ""}
                 for word in ("amber", "blue", "clear", "dreamy", "fresh")]
        result = style_candidates(values, words)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(candidate["direction_conflict"] for candidate in result))

    def test_p90_control_normalization_and_bounded_optimizer(self):
        config = direct_config()
        config["control_bounds"] = [list(pair) for pair in CONTROL_BOUNDS]
        raw_neutral = neutral_raw_controls(dtype=torch.float64, bounds=config["control_bounds"])
        deltas = torch.zeros((12, 6), dtype=torch.float64)
        for index in range(6):
            deltas[index, index] = .4
            deltas[index + 6, index] = -.3
        config["learned_calibration"] = {"raw_neutral": raw_neutral.tolist(),
                                          "pole_raw_deltas": deltas.tolist()}
        endpoints = pole_endpoints(config)
        for value, expected in ((-1., endpoints[0]), (0., endpoints[1]), (1., endpoints[2])):
            normalized = torch.full((6,), value, dtype=torch.float64)
            physical = normalized_to_physical(normalized, endpoints)
            torch.testing.assert_close(physical, expected, atol=1e-12, rtol=0)
            torch.testing.assert_close(physical_to_normalized(physical, endpoints), normalized, atol=1e-12, rtol=0)
        self.assertTrue(torch.all(physical_to_normalized(endpoints[2] + .01, endpoints) > 1))
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            image = torch.rand((1, 3, 24, 24), generator=torch.Generator().manual_seed(7)) * .6 + .1
            text = F.normalize(torch.tensor([[1., 2., 3., 1.]]), dim=-1)
            quality = F.normalize(torch.tensor([[1., 0., 0., 1.], [0., 1., 0., 1.],
                                                [0., 0., 1., 1.], [1., 1., 0., 0.]]), dim=-1)
            result = optimize(image, ToyModel(), ToyModel(), text, quality, config,
                              iterations=2, parameterization="p90")
            normalized = torch.tensor([result["normalized_controls"][name] for name in CONTROL_NAMES])
            physical = torch.tensor([result["controls"][name] for name in CONTROL_NAMES])
            self.assertTrue(torch.all(normalized.abs() <= 1 + 1e-6))
            torch.testing.assert_close(normalized, physical_to_normalized(physical, endpoints), atol=1e-6, rtol=0)
        finally:
            torch.set_num_threads(threads)

    def test_fivek_png_preserves_16bit_linear_rgb_and_aspect(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "linear.png"
            bgr = np.full((6, 12, 3), [1024, 32768, 65535], dtype=np.uint16)
            self.assertTrue(cv2.imwrite(str(path), bgr))
            image = load_linear_png(path)
            torch.testing.assert_close(image[0, :, 0, 0],
                torch.tensor([65535., 32768., 1024.]) / 65535.)
            resized = resize_short_side(image, 3)
            self.assertEqual(tuple(resized.shape), (1, 3, 3, 6))
            torch.testing.assert_close(resized[0, :, 0, 0], image[0, :, 0, 0])

    def test_agent_estimates_cover_all_categories_and_control_ranges(self):
        path = Path(__file__).with_name("sentences_agent_estimates.json")
        data = json.loads(path.read_text())
        rows = data["sentences"]
        controls = {"warmth", "saturation", "brightness", "contrast", "sharpness", "palette"}
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertEqual(len({row["text"].strip().casefold() for row in rows}), len(rows))
        for group in ("D1", "D2", "D3", "D4"):
            self.assertEqual(sum(row["group"] == group for row in rows), 18)
        self.assertEqual({row["ambiguity"] for row in rows}, {"direct", "low", "medium", "high"})
        for row in rows:
            self.assertTrue(row["id"].startswith(row["group"] + "_"))
            self.assertEqual(set(row["ratings"]), controls)
            self.assertTrue(all(type(value) in (int, float) and np.isfinite(value) and -1 <= value <= 1
                                for value in row["ratings"].values()))
        for control in controls:
            values = [row["ratings"][control] for row in rows]
            self.assertTrue(any(value < 0 for value in values))
            self.assertTrue(any(value > 0 for value in values))
            self.assertIn(0, values)

    def test_quality_pairs_and_signed_improvement(self):
        image_feature = torch.tensor([[1., 0.]])
        text_features = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])
        scores = paired_quality_scores(image_feature, text_features)
        torch.testing.assert_close(scores, torch.tensor([[1., -1.]]))
        terms = objective_terms(torch.tensor([.6]), scores, torch.tensor([[.8, -.7]]),
                                torch.tensor([1.]), torch.tensor([0.]))
        torch.testing.assert_close(terms["total"], torch.tensor(-.497), atol=1e-6, rtol=0)
        self.assertLess(float(terms["realism"]), 0.)
        self.assertGreater(float(terms["beauty"]), 0.)

    def test_loss_trace_and_named_warm_start(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            image = torch.rand((1, 3, 24, 24), generator=torch.Generator().manual_seed(7)) * .6 + .1
            text = F.normalize(torch.tensor([[1., 2., 3., 1.]]), dim=-1)
            quality = F.normalize(torch.tensor([[1., 0., 0., 1.], [0., 1., 0., 1.],
                                                [0., 0., 1., 1.], [1., 1., 0., 0.]]), dim=-1)
            criteria = {"window": 3, "min_iterations": 6, "loss_tolerance": 1e-12,
                        "control_tolerance": 1e-12, "gradient_tolerance": 1e-12}
            initial = [.3, -.2, .1, .2, -.1, .4]
            result = optimize(image, ToyModel(), ToyModel(), text, quality, direct_config(),
                              iterations=3, palette_starts=(0.,), convergence=criteria,
                              realism_weight=3., beauty_weight=3.,
                              additional_raw_starts=[{"name": "candidate", "raw_controls": initial}])
            restarts = result["convergence"]["restarts"]
            self.assertEqual(len(restarts), 2)
            self.assertEqual(restarts[1]["start_id"], "candidate")
            self.assertLessEqual(result["losses"]["total"], restarts[1]["trace"][0]["total"] + 1e-7)
            self.assertFalse(result["convergence"]["all_restarts_converged"])
            for restart in restarts:
                for row in restart["trace"]:
                    total = row["semantic"] + 3 * row["realism"] + 3 * row["beauty"] + row["regularization"]
                    self.assertAlmostEqual(row["total"], total, places=6)
                    self.assertEqual(len(row["quality_scores"]), 2)
        finally:
            torch.set_num_threads(threads)

    def test_six_control_multistart_keeps_neutral_candidate(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            image = torch.rand((1, 3, 24, 24), generator=torch.Generator().manual_seed(7)) * .6 + .1
            text = F.normalize(torch.tensor([[1., 2., 3., 1.]]), dim=-1)
            quality = F.normalize(torch.tensor([[1., 0., 0., 1.], [0., 1., 0., 1.],
                                                [0., 0., 1., 1.], [1., 1., 0., 0.]]), dim=-1)
            config = direct_config()
            neutral_only = optimize(image, ToyModel(), ToyModel(), text, quality, config,
                                    iterations=2, palette_starts=(0.,))
            multistart = optimize(image, ToyModel(), ToyModel(), text, quality, config, iterations=2)
            self.assertEqual(set(multistart["controls"]), set(CONTROL_NAMES))
            self.assertLessEqual(multistart["losses"]["total"], neutral_only["losses"]["total"] + 1e-7)
            self.assertGreater(abs(multistart["controls"]["palette"]), 1e-6)
            with self.assertRaisesRegex(ValueError, "include neutral"):
                optimize(image, ToyModel(), ToyModel(), text, quality, config,
                         iterations=1, palette_starts=(.02,))
        finally:
            torch.set_num_threads(threads)

    def test_sentence_splits_reject_duplicate_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sentences.json"
            path.write_text(json.dumps({"sentences": [
                {"id": "first", "text": "A warm photograph", "split": "train"},
                {"id": "second", "text": "a warm photograph", "split": "test"},
                {"id": "third", "text": "A cool photograph", "split": "validation"}]}))
            with self.assertRaisesRegex(ValueError, "unique across splits"):
                validate_sentences(path)

    def test_dataset_rejects_split_leakage_and_duplicate_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(json.dumps({"control_bounds": [[-.3, .3], [-1, 1], [.05, 5],
                                                               [.2, 3], [-.9, .9], [-.4, .4]]}))
            (root / "manifest.json").write_text(json.dumps({"config": str(config)}))
            np.savez_compressed(root / "text_embeddings.npz", ids=np.asarray(["example"]),
                                embeddings=np.asarray([[1., 0.]], dtype=float))
            controls = dict(zip(CONTROL_NAMES, (0., 0., 1., 1., 0., 0.)))
            row = {"sentence_id": "example", "sentence_split": "train", "image_id": 1,
                   "image_split": "train", "controls": controls}
            (root / "dataset.json").write_text(json.dumps({"records": [row, row]}))
            with self.assertRaisesRegex(ValueError, "Duplicate sentence-image pairs"):
                load_dataset(root)
            other = {**row, "image_id": 2, "sentence_split": "test"}
            (root / "dataset.json").write_text(json.dumps({"records": [row, other]}))
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                load_dataset(root)

    def test_p90_dataset_uses_normalized_targets_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoints = {name: {"negative": -2., "neutral": 0., "positive": 4.}
                         for name in CONTROL_NAMES}
            (root / "manifest.json").write_text(json.dumps({
                "parameterization": "p90", "pole_endpoints": endpoints}))
            np.savez_compressed(root / "text_embeddings.npz", ids=np.asarray(["example"]),
                                embeddings=np.asarray([[1., 0.]], dtype=float))
            row = {"sentence_id": "example", "sentence_split": "train", "image_id": 1,
                   "image_split": "train", "controls": {name: 2. for name in CONTROL_NAMES},
                   "normalized_controls": {name: .5 for name in CONTROL_NAMES}}
            (root / "dataset.json").write_text(json.dumps({"records": [row]}))
            _, _, _, targets, bounds = load_dataset(root)
            np.testing.assert_array_equal(targets, np.full((1, 6), .5))
            np.testing.assert_array_equal(bounds, np.tile([[-1., 1.]], (6, 1)))
            row["normalized_controls"]["warmth"] = -.5
            (root / "dataset.json").write_text(json.dumps({"records": [row]}))
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_dataset(root)

    def test_generator_saves_siglip_p90_records(self):
        class TextModel(torch.nn.Module):
            def encode_text(self, tokens):
                return torch.ones((len(tokens), 4), dtype=torch.float32)

        def fake_optimizer(image, semantic_model, quality_model, text_feature, quality_features, config, **kwargs):
            self.assertIs(semantic_model, quality_model)
            self.assertEqual(config["image_mean"], [.5, .5, .5])
            self.assertEqual(config["image_std"], [.5, .5, .5])
            self.assertEqual(kwargs["parameterization"], "p90")
            return {"controls": dict(zip(CONTROL_NAMES, (0., 0., 1., 1., 0., 0.))),
                    "normalized_controls": dict.fromkeys(CONTROL_NAMES, 0.), "parameterization": "p90",
                    "losses": {"total": 0.}}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = direct_config()
            config["control_bounds"] = [list(pair) for pair in CONTROL_BOUNDS]
            raw_neutral = neutral_raw_controls(dtype=torch.float64, bounds=config["control_bounds"])
            deltas = torch.zeros((12, 6), dtype=torch.float64)
            for index in range(6):
                deltas[index, index] = .4
                deltas[index + 6, index] = -.3
            config["learned_calibration"] = {"raw_neutral": raw_neutral.tolist(),
                                              "pole_raw_deltas": deltas.tolist()}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            sentence_path = root / "sentences.json"
            sentence_path.write_text(json.dumps({"sentences": [
                {"id": "train", "text": "A warmer image", "split": "train"},
                {"id": "validation", "text": "A cooler image", "split": "validation"},
                {"id": "test", "text": "A brighter image", "split": "test"}]}))
            data = root / "data"
            data.mkdir()
            for image_id in (1, 2):
                (data / f"{image_id}.dng").touch()
            args = SimpleNamespace(iters=1, size=24, lr=.03, train_images=[1], test_images=[2],
                data=data, sentences=sentence_path, config=config_path, out=root / "result",
                device="cpu", parameterization="p90", semantic_backbone="siglip")
            with patch("generate_dataset.open_clip.create_model_and_transforms", return_value=(TextModel(), None, None)) as factory, \
                    patch("generate_dataset.open_clip.get_tokenizer", return_value=lambda texts: torch.zeros((len(texts), 2), dtype=torch.long)), \
                    patch("generate_dataset.load_and_preprocess_raw", return_value=torch.ones((1, 3, 24, 24)) * .2), \
                    patch("generate_dataset.optimize", side_effect=fake_optimizer):
                generate_dataset(args)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(factory.call_args.args[0], "ViT-B-16-SigLIP")
            manifest, records, _, targets, bounds = load_dataset(args.out)
            self.assertEqual(manifest["semantic_backbone"], "siglip")
            self.assertEqual(manifest["parameterization"], "p90")
            self.assertEqual(len(records), 6)
            np.testing.assert_array_equal(targets, np.zeros((6, 6)))
            np.testing.assert_array_equal(bounds, np.tile([[-1., 1.]], (6, 1)))


if __name__ == "__main__":
    unittest.main()