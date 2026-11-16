#!/bin/bash
# Run the current main (or EXPERIMENT override) on 10 images x full word corpus.
# Results: /data/data1/semantic_isp/results/exp14_ten_images/<image>/{grid.png,results.json,log.txt}
set -euo pipefail
PY=/home/administrator/miniforge3/bin/python3
DIR=/data/data1/EmotionISP_dataset/dng_filtered
OUT=${OUT:-/data/data1/semantic_isp/results/exp14_ten_images}
HERE=$(cd "$(dirname "$0")" && pwd)
PER_GPU=${PER_GPU:-2}
EXPERIMENT=${EXPERIMENT:-exp07_emovis_diff.py}
if ! [[ "$PER_GPU" =~ ^[1-9][0-9]*$ ]] || [ ! -f "$HERE/$EXPERIMENT" ]; then
  echo "Invalid PER_GPU or missing experiment: $HERE/$EXPERIMENT" >&2
  exit 1
fi

# 1=desert 28=dog 2=child 236=man 194=swan 277=snow 263=dusk 384=backlit 33=indoor 103=city
# IMAGES="..." overrides the list (split alternately between GPU0 and GPU1).
IMAGES=${IMAGES:-"1 28 2 236 194 277 263 384 33 103"}
GPU0=""; GPU1=""; i=0
for n in $IMAGES; do
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "Invalid image ID: $n" >&2
    exit 1
  fi
  if (( i % 2 == 0 )); then GPU0="$GPU0 $n"; else GPU1="$GPU1 $n"; fi
  i=$((i+1))
done

run_image() {
  local gpu=$1 n=$2
  mkdir -p "$OUT/img_$n"
  if CUDA_VISIBLE_DEVICES="$gpu" DNG_PATH="$DIR/$n.dng" RESULTS_DIR="$OUT/img_$n" \
      "$PY" "$HERE/$EXPERIMENT" > "$OUT/img_$n/log.txt" 2>&1; then
    echo "done $n"
  else
    local status=$?
    echo "FAILED image $n on GPU $gpu (exit $status): $OUT/img_$n/log.txt" >&2
    return "$status"
  fi
}
export -f run_image
export PY DIR OUT HERE EXPERIMENT

run_queue() {
  local gpu=$1; shift
  printf '%s\n' "$@" | xargs -r -P "$PER_GPU" -I{} bash -c 'run_image "$1" "$2"' _ "$gpu" {}
}

mkdir -p "$OUT"
pids=()
if [ -n "$GPU0" ]; then run_queue 0 $GPU0 & pids+=("$!"); fi
if [ -n "$GPU1" ]; then run_queue 1 $GPU1 & pids+=("$!"); fi
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
