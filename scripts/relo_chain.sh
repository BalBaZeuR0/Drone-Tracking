#!/usr/bin/env bash
# RELO hibrit dedektörüyle tam önbellekler + C / fold B bölünmelerinde RL karşılaştırması.
# C_relo_ozelliksiz: RELO önbelleği ama RELO özellikleri gözlemde yok (kazanç konumdan mı, güvenden mi?).
# 5060 masaüstünde çalışır: bash scripts/relo_chain.sh  (durum: work/relo_chain_status.txt)
set -uo pipefail
R=/media/noron/DISK02/DroneRL
D=/media/noron/DISK02/FlyVision_Adapter/data/drone-tracking-datasets
C=$R/caches
STATUS=$R/work/relo_chain_status.txt
cd $R/DroneTrackingRL
status() { echo "[$(date '+%F %T')] $*" | tee -a $STATUS; }

build() { # ds son kameralar...
  ds=$1; end=$2; shift 2
  out=$C/${ds}_relo.npz
  if [ -f $out ]; then status "$out hazır, atlanıyor"; return; fi
  status "önbellek başlıyor: $ds 1-$end"
  OMP_NUM_THREADS=4 $R/.venv-relo/bin/python -m dronetrackingrl.real_data.experiment build-cache \
    --frames-root $R/frames/$ds --detections-dir $D/$ds/detections --ref-start 1 --ref-end $end \
    --sync $ds --cameras "$@" --workers $# --parts-dir $R/work/relo_full_parts \
    --detector relo --relo-root $R/RELO --out $out > $R/work/build_${ds}_relo.log 2>&1 \
    || { status "HATA: $ds önbellek"; exit 1; }
  status "önbellek bitti: $out"
}

build dataset3 33875 0 1 2 3 4 5
build dataset4 31075 0 1 2 3 4 5 6

status "adil dedektör karşılaştırması (ikisi de 1. kareden)"
for ds in dataset3 dataset4; do
  $R/.venv-relo/bin/python scripts/relo_gate.py $C/${ds}_fixed.npz $C/${ds}_relo.npz 2>&1 | grep -v Warn \
    > $R/work/gate_full_${ds}.txt
  tail -2 $R/work/gate_full_${ds}.txt | tee -a $STATUS
done

COMMON="--timesteps 200000 --seeds 0 1 2 --latent-dim 16 --encoder-epochs 5 --switch-penalty 0.1 --include-own-history --random-seeds 5"
c_split() { # ad ds3önbellek ds4önbellek gözlem
  echo "--train-cache $2 --eval-cache $2 --train-range 1 11800 --eval-range 12001 14000 \
    --extra-train $2:14200:27800 $2:30201:33875 $3:1:4800 $3:10201:15800 $3:18201:26000 \
    --extra-eval $2:28001:30000 $3:5001:10000 $3:16001:18000 --obs-mode $4 --out-dir $R/outputs/$1"
}
b_split() {
  echo "--train-cache $2 --eval-cache $2 --train-range 12001 14000 --eval-range 1 11800 \
    --extra-train $2:28001:30000 $3:5001:10000 $3:16001:18000 \
    --extra-eval $2:14200:27800 $2:30201:33875 $3:1:4800 $3:10201:15800 $3:18201:26000 --obs-mode $4 --out-dir $R/outputs/$1"
}
train() { # ad argümanlar...
  name=$1; shift
  if [ -f $R/outputs/$name/results.json ]; then status "$name hazır"; return; fi
  OMP_NUM_THREADS=7 MKL_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 TORCH_NUM_THREADS=7 \
    $R/DroneTrackingRL/.venv/bin/python -m dronetrackingrl.real_data.experiment train $COMMON "$@" \
    > $R/work/train_${name}.log 2>&1 && status "eğitim bitti: $name" || status "HATA: eğitim $name"
}

status "eğitimler başlıyor (4 paralel)"
train C_relo $(c_split C_relo $C/dataset3_relo.npz $C/dataset4_relo.npz latent+aux+relo) &
train C_fixed $(c_split C_fixed $C/dataset3_fixed.npz $C/dataset4_fixed.npz latent+aux) &
train C_relo_ozelliksiz $(c_split C_relo_ozelliksiz $C/dataset3_relo.npz $C/dataset4_relo.npz latent+aux) &
train foldB_relo $(b_split foldB_relo $C/dataset3_relo.npz $C/dataset4_relo.npz latent+aux+relo) &
wait
status "ZİNCİR BİTTİ"
