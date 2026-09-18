from pathlib import Path

import torch

from dronetrackingrl.real_data.training import train_and_evaluate

FRAMES_ROOT = Path("tests/fixtures/dataset3_sample/frames")
DETECTIONS_DIR = Path("tests/fixtures/dataset3_sample/detections")


def test_train_and_evaluate_runs_on_real_fixture_and_reports_full_visibility_reward():
    torch.manual_seed(0)
    avg_reward = train_and_evaluate(
        frames_root=FRAMES_ROOT,
        detections_dir=DETECTIONS_DIR,
        cameras=[0],
        start_ref_frame=1,
        end_ref_frame=3,
        reference_camera=0,
        total_timesteps=200,
        seed=0,
    )
    # Tek kamera, 3 gerçek fixture karesinin (1,2,3) hepsi ground truth'a
    # göre görünür -- ödül politikadan bağımsız her zaman +1, yani bu bir
    # kablolama (wiring) duman testi, öğrenme testi değil (PPO'nun gerçekten
    # rastgeleyi yendiğinin kanıtı sentetik Task 7'de zaten var).
    assert avg_reward == 1.0
