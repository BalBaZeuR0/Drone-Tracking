#!/usr/bin/env bash
# Saf yeni sahnede RELO özellikleri mi, RELO konumu mu? (bkz. EXPERIMENTS.md "RELO kazancının tekrarı")
# RELO önbelleği + latent+aux (güven/kutu/bayrak yok); X_*_fixed ve X_*_relo ile aynı aralık ve tohumlar.
# 5060 masaüstünde: bash scripts/relo_ablation.sh   (durum: work/relo_ablation_status.txt)
set -uo pipefail
R=/media/noron/DISK02/DroneRL
C=$R/caches
STATUS=$R/work/relo_ablation_status.txt
cd $R/DroneTrackingRL
status() { echo "[$(date '+%F %T')] $*" | tee -a $STATUS; }

for f in dataset3_relo dataset4_relo; do
  [ -f $C/$f.npz ] || { status "HATA: $C/$f.npz yok"; exit 1; }
done

BASE="--timesteps 200000 --seeds 0 1 2 --latent-dim 16 --encoder-epochs 5 --switch-penalty 0.1 --include-own-history --random-seeds 5"
train() { # ad eğitim-önbelleği eğitim-sonu test-önbelleği test-sonu
  name=$1
  if [ -f $R/outputs/$name/results.json ]; then status "$name hazır"; return; fi
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TORCH_NUM_THREADS=8 \
    $R/DroneTrackingRL/.venv/bin/python -m dronetrackingrl.real_data.experiment train $BASE \
    --train-cache $2 --train-range 1 $3 --eval-cache $4 --eval-range 1 $5 --obs-mode latent+aux \
    --out-dir $R/outputs/$name > $R/work/train_${name}.log 2>&1 && status "eğitim bitti: $name" || status "HATA: eğitim $name"
}

status "2 eğitim başlıyor (paralel)"
train X_ds3_to_ds4_relo_ozelliksiz $C/dataset3_relo.npz 33875 $C/dataset4_relo.npz 26000 &
train X_ds4_to_ds3_relo_ozelliksiz $C/dataset4_relo.npz 26000 $C/dataset3_relo.npz 33875 &
wait
status "ABLASYON BİTTİ"
