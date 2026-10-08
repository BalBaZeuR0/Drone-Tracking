#!/usr/bin/env bash
set -uo pipefail
R=/media/noron/DISK02/DroneRL
D=/media/noron/DISK02/FlyVision_Adapter/data/drone-tracking-datasets
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $R/DroneTrackingRL
run() { # ds start end cams...
  ds=$1; a=$2; b=$3; shift 3
  out=$R/caches/relo_gate_${ds}_${a}_${b}.npz
  [ -f $out ] || $R/.venv-relo/bin/python -m dronetrackingrl.real_data.experiment build-cache \
    --frames-root $R/frames/$ds --detections-dir $D/$ds/detections --ref-start $a --ref-end $b \
    --sync $ds --cameras "$@" --workers $# --parts-dir $R/work/relo_parts --detector relo --relo-root $R/RELO --out $out
  echo "=== $ds $a-$b ==="
  $R/.venv-relo/bin/python $R/work/relo_gate.py $R/caches/${ds}_fixed.npz $out
}
run dataset4 5001 7000 0 1 2 3 4 5 6
run dataset4 16001 18000 0 1 2 3 4 5 6
run dataset3 12001 14000 0 1 2 3 4 5
echo GATE_DONE
