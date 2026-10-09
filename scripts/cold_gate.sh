#!/usr/bin/env bash
set -uo pipefail
R=/media/noron/DISK02/DroneRL
D=/media/noron/DISK02/FlyVision_Adapter/data/drone-tracking-datasets
export OMP_NUM_THREADS=4
cd $R/DroneTrackingRL
for spec in "dataset4 5001 7000 0 1 2 3 4 5 6" "dataset4 16001 18000 0 1 2 3 4 5 6" "dataset3 12001 14000 0 1 2 3 4 5" "dataset3 20001 21500 0 1 2 3 4 5"; do
  set -- $spec; ds=$1; a=$2; b=$3; shift 3
  out=$R/caches/cold_classic_${ds}_${a}_${b}.npz
  [ -f $out ] || $R/.venv-relo/bin/python -m dronetrackingrl.real_data.experiment build-cache --frames-root $R/frames/$ds \
     --detections-dir $D/$ds/detections --ref-start $a --ref-end $b --sync $ds --cameras "$@" --workers $# \
     --parts-dir $R/work/cold_parts --out $out > /dev/null 2>&1
  relo=$R/caches/relo_gate_${ds}_${a}_${b}.npz; [ -f $relo ] || relo=$R/caches/relo_pilot2_ds3_20001_21500.npz
  echo "=== $ds $a-$b (ikisi de soğuk başlangıç) ==="
  $R/.venv-relo/bin/python $R/work/relo_gate.py $out $relo 2>&1 | grep -v Warn
done
echo COLD_DONE
