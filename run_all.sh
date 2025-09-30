#!/usr/bin/env bash
set -euo pipefail

BASE="../Datasets/waymo/dynamic"
PROC="$BASE/processed"
COLMAP="$BASE/colmap"

OUT_BASE="./output"
RUN="uv run python train.py"
R=4   # -r 값

i=1
for seg_path in "$PROC"/*_with_camera_labels; do
  seg_name="$(basename "$seg_path")"
  colmap_path="$COLMAP/$seg_name"
  out_dir="$OUT_BASE/test7-$i"

  echo "[$i] Training on: $seg_name"
  echo "    -s $seg_path"
  echo "    -c $colmap_path"
  echo "    -m $out_dir"
  mkdir -p "$out_dir"

  # 실제 실행 + 로그 저장
  $RUN -s "$seg_path" -c "$colmap_path" -m "$out_dir" -r "$R" |& tee "$out_dir/train.log"

  i=$((i+1))
done

echo "Done. Outputs under $OUT_BASE/test7-*"
