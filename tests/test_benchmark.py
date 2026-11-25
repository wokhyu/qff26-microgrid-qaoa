from microgrid.benchmark import main, read_csv


def test_benchmark_cli_writes_results_and_replots(tmp_path):
    main(["--n", "6", "--density", "0.3", "--seeds", "0", "--max-p", "1", "--shots", "200", "--out", str(tmp_path)])
    rows = read_csv(tmp_path / "benchmark.csv")
    methods = {r["method"] for r in rows}
    assert methods == {"exact", "greedy", "sa", "qaoa_ideal"}
    assert all(r["is_optimal"] == 1 for r in rows if r["method"] == "exact")
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith("## Solution quality")
    assert len(list((tmp_path / "figures").glob("*.png"))) == 3  # no depth plot without --transpile

    main(["--replot", "--out", str(tmp_path)])
    assert (tmp_path / "figures" / "baselines.png").exists()
