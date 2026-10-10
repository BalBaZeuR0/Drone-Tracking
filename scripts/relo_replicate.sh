#!/usr/bin/env bash
# RELO'nun fold B kazancı tekrarlanıyor mu? (bkz. EXPERIMENTS.md "RELO hibrit dedektörle RL")
#  1) fold B, yeni tohumlar (3 4 5): foldB_mix_s345 (klasik) vs foldB_relo_s345 (RELO)
#  2) saf yeni sahne: dataset3 tamamında eğit -> dataset4 1-26000 test (A2 ile aynı aralık), tersi: dataset4 1-26000 eğit (B2 ile aynı) -> dataset3 tamamı; klasik vs RELO
# Önbellekler hazır olmalı: caches/dataset{3,4}_{fixed,relo}.npz (scripts/relo_chain.sh üretir).
# 5060 masaüstünde: bash scripts/relo_replicate.sh   (durum: work/relo_replicate_status.txt)
set -uo pipefail
R=/media/noron/DISK02/DroneRL
C=$R/caches
STATUS=$R/work/relo_replicate_status.txt
cd $R/DroneTrackingRL
status() { echo "[$(date '+%F %T')] $*" | tee -a $STATUS; }

for f in dataset3_fixed dataset4_fixed dataset3_relo dataset4_relo; do
  [ -f $C/$f.npz ] || { status "HATA: $C/$f.npz yok"; exit 1; }
done

BASE="--timesteps 200000 --latent-dim 16 --encoder-epochs 5 --switch-penalty 0.1 --include-own-history --random-seeds 5"
b_split() { # ad ds3 ds4 gözlem
  echo "--train-cache $2 --eval-cache $2 --train-range 12001 14000 --eval-range 1 11800 \
    --extra-train $2:28001:30000 $3:5001:10000 $3:16001:18000 \
    --extra-eval $2:14200:27800 $2:30201:33875 $3:1:4800 $3:10201:15800 $3:18201:26000 --obs-mode $4 --out-dir $R/outputs/$1"
}
cross() { # ad eğitim-önbelleği eğitim-sonu test-önbelleği test-sonu gözlem
  echo "--train-cache $2 --train-range 1 $3 --eval-cache $4 --eval-range 1 $5 --obs-mode $6 --out-dir $R/outputs/$1"
}
train() { # ad tohumlar argümanlar...
  name=$1; seeds=$2; shift 2
  if [ -f $R/outputs/$name/results.json ]; then status "$name hazır"; return; fi
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4 \
    $R/DroneTrackingRL/.venv/bin/python -m dronetrackingrl.real_data.experiment train $BASE --seeds $seeds "$@" \
    > $R/work/train_${name}.log 2>&1 && status "eğitim bitti: $name" || status "HATA: eğitim $name"
}

status "6 eğitim başlıyor (paralel)"
train foldB_mix_s345 "3 4 5" $(b_split foldB_mix_s345 $C/dataset3_fixed.npz $C/dataset4_fixed.npz latent+aux) &
train foldB_relo_s345 "3 4 5" $(b_split foldB_relo_s345 $C/dataset3_relo.npz $C/dataset4_relo.npz latent+aux+relo) &
train X_ds3_to_ds4_fixed "0 1 2" $(cross X_ds3_to_ds4_fixed $C/dataset3_fixed.npz 33875 $C/dataset4_fixed.npz 26000 latent+aux) &
train X_ds3_to_ds4_relo "0 1 2" $(cross X_ds3_to_ds4_relo $C/dataset3_relo.npz 33875 $C/dataset4_relo.npz 26000 latent+aux+relo) &
train X_ds4_to_ds3_fixed "0 1 2" $(cross X_ds4_to_ds3_fixed $C/dataset4_fixed.npz 26000 $C/dataset3_fixed.npz 33875 latent+aux) &
train X_ds4_to_ds3_relo "0 1 2" $(cross X_ds4_to_ds3_relo $C/dataset4_relo.npz 26000 $C/dataset3_relo.npz 33875 latent+aux+relo) &
wait
status "TEKRAR ZİNCİRİ BİTTİ"
