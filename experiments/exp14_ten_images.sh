#!/bin/bash
# Experiment 14: run exp07 (current main, unchanged ISP) on 10 diverse images x full word corpus.
# Results: /data/data1/semantic_isp/results/exp14_ten_images/<image>/{grid.png,results.json,log.txt}
PY=/home/administrator/miniforge3/bin/python3
DIR=/data/data1/EmotionISP_dataset/dng_filtered
OUT=${OUT:-/data/data1/semantic_isp/results/exp14_ten_images}
HERE=$(cd "$(dirname "$0")" && pwd)
PER_GPU=${PER_GPU:-2}

# 1=desert 28=dog 2=child 236=man 194=swan 277=snow 263=dusk 384=backlit 33=indoor 103=city
# IMAGES="..." overrides the list (split alternately between GPU0 and GPU1).
IMAGES=${IMAGES:-"1 28 2 236 194 277 263 384 33 103"}
GPU0=""; GPU1=""; i=0
for n in $IMAGES; do
  if (( i % 2 == 0 )); then GPU0="$GPU0 $n"; else GPU1="$GPU1 $n"; fi
  i=$((i+1))
done

run_queue() {  # $1=gpu id, rest=images
  local gpu=$1; shift
  for n in "$@"; do echo $n; done | xargs -P "$PER_GPU" -I{} bash -c '
    mkdir -p '"$OUT"'/img_{}
    CUDA_VISIBLE_DEVICES='"$gpu"' DNG_PATH='"$DIR"'/{}.dng RESULTS_DIR='"$OUT"'/img_{} \
      '"$PY"' '"$HERE"'/exp07_emovis_diff.py > '"$OUT"'/img_{}/log.txt 2>&1
    echo "done {}"
  '
}

mkdir -p "$OUT"
[ -n "$GPU0" ] && run_queue 0 $GPU0 &
[ -n "$GPU1" ] && run_queue 1 $GPU1 &
wait
