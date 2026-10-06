import json

from dronetrackingrl.real_data.ledger import add_run, read_ledger, rows_from_results, table_markdown


def _fake_results():
    split = {"ref_start": 1, "ref_end": 100, "steps": 99, "cameras": [0, 1]}
    baseline = {
        "random": {"visible_rate_mean": 0.5, "visible_rate_std": 0.01, "runs": 3},
        "fixed": {},
        "best_fixed": {"camera": 1, "visible_rate": 0.8},
        "oracle": {"visible_rate": 0.95},
    }
    ppo = {
        "0": {"splits": {"eval": {"visible_rate": 0.9, "switches": 10}}},
        "1": {"splits": {"eval": {"visible_rate": 0.8, "switches": 20}}},
    }
    return {
        "config": {"train_cache": "/x/dataset4.npz", "eval_cache": "/x/dataset4.npz", "obs_mode": "latent+aux",
                   "switch_penalty": 0.1, "include_own_history": True},
        "splits": {"eval": split},
        "baselines": {"eval": baseline},
        "ppo": ppo,
    }


def test_rows_summarize_one_split_across_seeds():
    row = rows_from_results(_fake_results(), "demo")[0]
    assert row["name"] == "demo" and row["split"] == "eval"
    assert row["ppo_mean"] == 0.85 and row["seeds"] == 2
    assert row["ppo_seed_rates"] == "0.9000;0.8000"
    assert row["switches_mean"] == 15.0
    assert row["best_fixed_cam"] == 1 and row["oracle"] == 0.95
    assert row["include_own_history"] is True and row["train_cache"] == "dataset4.npz"


def test_add_run_replaces_same_name_and_keeps_others(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.json").write_text(json.dumps(_fake_results()), encoding="utf-8")
    results_dir = tmp_path / "results"
    add_run(run, "a", results_dir)
    add_run(run, "b", results_dir)
    add_run(run, "a", results_dir)  # aynı ad: satır değişir, çoğalmaz
    rows = read_ledger(results_dir / "ledger.csv")
    assert sorted(r["name"] for r in rows) == ["a", "b"]
    assert (results_dir / "a" / "results.json").exists()
    assert "| a |" in table_markdown(rows, ["a"])
