"""Benchmark QAOA against classical baselines over n, p and feeder density.

Example:

    python -m microgrid.benchmark --n 6 8 10 12 --density 0.2 0.4 0.6 --seeds 0 1 2 --max-p 3 --noisy

Writes results/benchmark.csv, results/summary.md and results/figures/*.png.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .baselines import brute_force, greedy_merit_order, simulated_annealing
from .instance import make_microgrid
from .metrics import circuit_stats, dispatch_report, probability_of_optimal
from .qaoa import SamplerEvaluator, build_ansatz, prepare, sample_best, solve_layerwise
from .qubo import build_qubo

FIELDS = [
    "n", "density", "edges", "couplings", "seed", "method", "p",
    "energy", "approx_ratio", "is_optimal", "gap_usd",
    "expectation", "exp_approx_ratio", "p_opt", "p_opt_random", "amplification",
    "cost_usd_per_h", "pct_demand_met", "unmet_kw", "surplus_kw",
    "runtime_s", "nfev", "depth", "two_qubit_gates", "two_qubit_depth",
]


def _row(base: dict, method: str, energy: float, x, grid, landscape, exact_e, **kw) -> dict:
    report = dispatch_report(grid, x)
    row = dict(base)
    row.update(
        method=method,
        energy=energy,
        approx_ratio=landscape.approximation_ratio(energy),
        is_optimal=int(np.isclose(energy, exact_e, rtol=1e-9, atol=1e-9)),
        gap_usd=energy - exact_e,
        cost_usd_per_h=report["cost_usd_per_h"],
        pct_demand_met=report["pct_demand_met"],
        unmet_kw=report["unmet_kw"],
        surplus_kw=report["surplus_kw"],
    )
    row.update(kw)
    return row


def run(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    backend = sim = None
    if args.noisy or args.transpile:
        from .hardware import fake_backend, noisy_simulator, transpile_for

        backend = fake_backend(args.backend)
        if args.noisy:
            sim = noisy_simulator(args.backend)

    transfer: dict[float, dict[int, np.ndarray]] = defaultdict(dict)
    for n in sorted(args.n):
        for density in args.density:
            for seed in args.seeds:
                t0 = time.perf_counter()
                grid = make_microgrid(n, density, seed)
                qubo = build_qubo(grid)
                problem = prepare(qubo)
                exact, land = brute_force(qubo)
                base = dict(n=n, density=density, edges=len(grid.edges), couplings=qubo.num_couplings(), seed=seed)
                p_rand = len(land.optimal_indices) / 2**n

                rows.append(_row(base, "exact", exact.energy, exact.x, grid, land, exact.energy,
                                 p=0, runtime_s=exact.runtime_s))
                greedy = greedy_merit_order(grid, qubo)
                rows.append(_row(base, "greedy", greedy.energy, greedy.x, grid, land, exact.energy,
                                 p=0, runtime_s=greedy.runtime_s))
                sa = simulated_annealing(qubo, seed=seed)
                rows.append(_row(base, "sa", sa.energy, sa.x, grid, land, exact.energy,
                                 p=0, runtime_s=sa.runtime_s))

                results = solve_layerwise(problem, args.max_p, objective=args.objective, alpha=args.alpha,
                                          maxiter=args.maxiter, transfer=transfer[density])
                for r in results:
                    transfer[density][r.reps] = r.theta
                    x, e = sample_best(problem, r.probs, args.shots, seed=seed)
                    p_opt = probability_of_optimal(r.probs, land.optimal_indices)
                    stats = {}
                    if backend is not None:
                        ansatz = build_ansatz(problem.normalized_hamiltonian, r.reps, measure=True)
                        isa = transpile_for(ansatz.circuit, backend)
                        stats = circuit_stats(isa)
                    rows.append(_row(
                        base, "qaoa_ideal", e, x, grid, land, exact.energy, p=r.reps,
                        expectation=r.expectation, exp_approx_ratio=land.approximation_ratio(r.expectation),
                        p_opt=p_opt, p_opt_random=p_rand, amplification=p_opt / p_rand,
                        runtime_s=r.runtime_s, nfev=r.nfev,
                        **{k: stats.get(k, "") for k in ("depth", "two_qubit_gates", "two_qubit_depth")},
                    ))
                    if sim is not None and seed in args.noisy_seeds:
                        from qiskit_ibm_runtime import SamplerV2

                        t1 = time.perf_counter()
                        evaluator = SamplerEvaluator(SamplerV2(mode=sim), ansatz, isa, shots=args.shots)
                        probs = evaluator.probabilities(r.theta)
                        exp_noisy = float(probs @ problem.energies)
                        sampled = np.flatnonzero(probs)
                        best = int(sampled[np.argmin(problem.energies[sampled])])
                        x_noisy = (best >> np.arange(n)) & 1
                        p_opt_noisy = probability_of_optimal(probs, land.optimal_indices)
                        rows.append(_row(
                            base, "qaoa_noisy", float(problem.energies[best]), x_noisy, grid, land, exact.energy,
                            p=r.reps, expectation=exp_noisy, exp_approx_ratio=land.approximation_ratio(exp_noisy),
                            p_opt=p_opt_noisy, p_opt_random=p_rand, amplification=p_opt_noisy / p_rand,
                            runtime_s=time.perf_counter() - t1,
                            **{k: stats.get(k, "") for k in ("depth", "two_qubit_gates", "two_qubit_depth")},
                        ))
                print(f"n={n:2d} density={density:.2f} seed={seed} edges={len(grid.edges):2d} "
                      f"couplings={qubo.num_couplings():2d} done in {time.perf_counter() - t0:.1f}s", flush=True)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def _mean(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows if r.get(key, "") != ""]
    return float(np.mean(values)) if values else float("nan")


def summarize(rows: list[dict], max_p: int) -> str:
    """Markdown tables averaged over seeds and densities."""
    by = defaultdict(list)
    for r in rows:
        by[(r["n"], r["method"], r["p"])].append(r)
    ns = sorted({r["n"] for r in rows})

    lines = [
        "## Solution quality (mean over instances)",
        "",
        "QAOA rows report the best of the sampled bitstrings (shots as configured). "
        "`objective gap` is the QUBO objective above the exact optimum: generation cost plus the "
        "balance penalty, so it is not a cash amount. `surplus` is supply above demand (reserve plus curtailment).",
        "",
        "| n | method | p | found optimum | objective gap | % demand met | surplus (kW) | generation cost (USD/h) | runtime (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    methods = [("exact", 0), ("greedy", 0), ("sa", 0), ("qaoa_ideal", max_p), ("qaoa_noisy", max_p)]
    for n in ns:
        for m, p in methods:
            group = by.get((n, m, p))
            if not group:
                continue
            lines.append(
                f"| {n} | {m} | {p or '-'} | {100 * _mean(group, 'is_optimal'):.0f}% | {_mean(group, 'gap_usd'):.2f} "
                f"| {_mean(group, 'pct_demand_met'):.1f} | {_mean(group, 'surplus_kw'):.1f} "
                f"| {_mean(group, 'cost_usd_per_h'):.2f} "
                f"| {_mean(group, 'runtime_s'):.2f} |"
            )

    lines += [
        "",
        "## How much QAOA concentrates probability on the optimum",
        "",
        "`P(opt)` is the probability that one shot returns an optimal dispatch. `random` is the same for a uniform guess. "
        "`amplification` = P(opt) / random. `expectation ratio` is the approximation ratio of the mean energy.",
        "",
        "| n | method | p | expectation ratio | P(opt) | random | amplification | 2q depth (transpiled) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in ns:
        for m in ("qaoa_ideal", "qaoa_noisy"):
            for p in range(1, max_p + 1):
                group = by.get((n, m, p))
                if not group:
                    continue
                lines.append(
                    f"| {n} | {m} | {p} | {_mean(group, 'exp_approx_ratio'):.3f} | {_mean(group, 'p_opt'):.4f} "
                    f"| {_mean(group, 'p_opt_random'):.5f} | {_mean(group, 'amplification'):.1f}x "
                    f"| {_mean(group, 'two_qubit_depth'):.0f} |"
                )

    def shots_95(p: float) -> float:
        """Shots needed for a 95% chance of sampling an optimal bitstring at least once."""
        return float("inf") if p <= 0 else (1.0 if p >= 1 else np.log(0.05) / np.log1p(-p))

    lines += [
        "",
        f"## Shots needed to be 95% sure of sampling the optimal plan (p = {max_p}, mean over instances)",
        "",
        "Best-of-shots success is easy at small n, because random guessing also gets lucky. "
        "This table is the fair comparison.",
        "",
        "| n | random guessing | QAOA noise-free | QAOA Heron noise model |",
        "|---|---|---|---|",
    ]
    for n in ns:
        ideal = by.get((n, "qaoa_ideal", max_p), [])
        noisy = by.get((n, "qaoa_noisy", max_p), [])
        if not ideal:
            continue
        rand = np.mean([shots_95(r["p_opt_random"]) for r in ideal])
        q_ideal = np.mean([shots_95(r["p_opt"]) for r in ideal])
        q_noisy = f"{np.mean([shots_95(r['p_opt']) for r in noisy]):,.0f}" if noisy else "-"
        lines.append(f"| {n} | {rand:,.0f} | {q_ideal:,.0f} | {q_noisy} |")
    return "\n".join(lines) + "\n"


# Chart styling: validated categorical palette (fixed order, never cycled), recessive axes,
# text in text tokens, direct labels on every line so identity is never colour alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"


def _style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": TEXT_2,
        "lines.linewidth": 2.0, "lines.markersize": 6.5, "font.family": "DejaVu Sans",
    })


def _end_labels(ax, series: list[tuple[list, list, str]], min_gap: float = 0.065) -> None:
    """Direct labels at the end of each line, nudged apart so they never overlap.

    ``min_gap`` is the minimum vertical spacing as a fraction of the y-axis span.
    Call after everything is plotted so the axis limits are final.
    """
    from matplotlib.transforms import offset_copy

    log = ax.get_yscale() == "log"
    fwd = np.log10 if log else (lambda v: v)
    inv = (lambda v: 10.0**v) if log else (lambda v: v)
    lo, hi = (fwd(v) for v in ax.get_ylim())
    gap = min_gap * (hi - lo)
    ends = []
    for xs, ys, text in series:
        pts = [(x, y) for x, y in zip(xs, ys) if not np.isnan(y) and (not log or y > 0)]
        if pts:
            ends.append([pts[-1][0], fwd(pts[-1][1]), text])
    ends.sort(key=lambda e: e[1])

    # Merge labels that would overlap into clusters; each cluster is centred on the mean of
    # its points, so labels stay as close as possible to the line ends they name.
    clusters: list[list] = []
    for e in ends:
        clusters.append([e])
        while len(clusters) > 1:
            prev, last = clusters[-2], clusters[-1]
            prev_top = np.mean([p[1] for p in prev]) + (len(prev) - 1) * gap / 2
            last_bottom = np.mean([p[1] for p in last]) - (len(last) - 1) * gap / 2
            if last_bottom - prev_top >= gap:
                break
            clusters[-2:] = [prev + last]
    shifted = offset_copy(ax.transData, fig=ax.figure, x=7, units="points")
    for cluster in clusters:
        centre = np.mean([p[1] for p in cluster])
        for k, (x, _, text) in enumerate(cluster):
            v = centre + (k - (len(cluster) - 1) / 2) * gap
            ax.text(x, inv(v), text, transform=shifted, va="center", fontsize=8.5, color=TEXT_2)


def plot(rows: list[dict], out: Path, max_p: int) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    out.mkdir(parents=True, exist_ok=True)
    ns = sorted({r["n"] for r in rows})
    densities = sorted({r["density"] for r in rows})
    color = {n: SERIES[k % len(SERIES)] for k, n in enumerate(ns)}
    has_noisy = any(r["method"] == "qaoa_noisy" for r in rows)
    paths: list[Path] = []

    def group(n=None, method=None, p=None, density=None):
        return [r for r in rows if (n is None or r["n"] == n) and (method is None or r["method"] == method)
                and (p is None or r["p"] == p) and (density is None or r["density"] == density)]

    def save(fig, name):
        fig.tight_layout()
        paths.append(out / name)
        fig.savefig(paths[-1], dpi=170)
        plt.close(fig)

    def save_with_legend(fig, ax, name):
        """Legend in one row under the panels, clear of every line."""
        handles, labs = ax.get_legend_handles_labels()
        fig.legend(handles, [f"{lab} qubits" for lab in labs], loc="lower center", ncol=len(labs),
                   bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        paths.append(out / name)
        fig.savefig(paths[-1], dpi=170)
        plt.close(fig)

    panels = [("qaoa_ideal", "Noise-free simulator")]
    if has_noisy:
        panels.append(("qaoa_noisy", "IBM Heron noise model"))

    # 1. Amplification of P(optimal) over random guessing, vs layers.
    ps = list(range(1, max_p + 1))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.9), sharey=True, squeeze=False)
    labelled = []
    for ax, (method, title) in zip(axes[0], panels):
        series = []
        for n in ns:
            ys = [_mean(group(n, method, p), "amplification") for p in ps]
            ax.plot(ps, ys, "o-", color=color[n], label=f"n = {n}")
            series.append((ps, ys, f"n={n}"))
        ax.axhline(1.0, color=TEXT_2, lw=1, ls=":")
        ax.annotate("random guess", (1, 1.0), xytext=(0, 4), textcoords="offset points", fontsize=8, color=TEXT_2)
        ax.set_yscale("log")
        ax.set_xticks(ps)
        ax.set_xlim(0.8, max_p + 0.45)
        ax.set_xlabel("QAOA layers p")
        ax.set_title(title)
        labelled.append((ax, series))
    for ax, series in labelled:
        _end_labels(ax, series)
    axes[0][0].set_ylabel("P(optimal plan) / P(random guess)")
    fig.suptitle("QAOA makes the optimal dispatch many times more likely than guessing",
                 fontsize=11.5, color=TEXT, weight="bold")
    save_with_legend(fig, axes[0][0], "amplification_vs_p.png")

    # 2. Transpiled two-qubit depth vs feeder density.
    if any(r.get("two_qubit_depth", "") != "" for r in rows):
        fig, ax = plt.subplots(figsize=(5.6, 3.9))
        series = []
        for n in ns:
            ys = [_mean(group(n, "qaoa_ideal", 1, d), "two_qubit_depth") for d in densities]
            ax.plot(densities, ys, "o-", color=color[n], label=f"n = {n}")
            series.append((densities, ys, f"n={n}"))
        ax.set_xticks(densities)
        ax.set_xlim(min(densities) - 0.03, max(densities) + 0.09)
        ax.set_ylim(bottom=0)
        _end_labels(ax, series)
        ax.set_xlabel("feeder edge density")
        ax.set_ylabel("two-qubit depth after routing (p = 1)")
        ax.set_title("Denser grids need deeper circuits on IBM Heron")
        ax.legend(loc="upper left", title="qubits", title_fontsize=8.5)
        save(fig, "depth_vs_density.png")

    # 3. Approximation ratio of <E> vs density at max p.
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.9), sharey=True, squeeze=False)
    labelled = []
    for ax, (method, title) in zip(axes[0], panels):
        series = []
        for n in ns:
            ys = [_mean(group(n, method, max_p, d), "exp_approx_ratio") for d in densities]
            ax.plot(densities, ys, "o-", color=color[n], label=f"n = {n}")
            series.append((densities, ys, f"n={n}"))
        ax.set_xticks(densities)
        ax.set_xlim(min(densities) - 0.03, max(densities) + 0.1)
        ax.set_xlabel("feeder edge density")
        ax.set_title(title)
        labelled.append((ax, series))
    for ax, series in labelled:
        _end_labels(ax, series)
    axes[0][0].set_ylabel(f"approximation ratio of mean energy (p = {max_p})")
    fig.suptitle("Mean QAOA solution quality across grid densities", fontsize=11.5, color=TEXT, weight="bold")
    save_with_legend(fig, axes[0][0], "ratio_vs_density.png")

    # 4. Methods side by side: how often each finds the optimum, and how much energy it wastes.
    methods = ["exact", "greedy", "sa", "qaoa_ideal", "qaoa_noisy"]
    labels = {"exact": "exact (brute force)", "greedy": "greedy merit-order", "sa": "simulated annealing",
              "qaoa_ideal": f"QAOA p={max_p}, noise-free", "qaoa_noisy": f"QAOA p={max_p}, Heron noise"}
    present = [m for m in methods if group(method=m)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    width = 0.8 / len(present)
    for k, m in enumerate(present):
        p = max_p if m.startswith("qaoa") else 0
        xs = np.arange(len(ns)) - 0.4 + (k + 0.5) * width
        kw = dict(width=width, color=SERIES[k], edgecolor=SURFACE, linewidth=2, label=labels[m], zorder=3)
        axes[0].bar(xs, [100 * _mean(group(n, m, p), "is_optimal") for n in ns], **kw)
        axes[1].bar(xs, [_mean(group(n, m, p), "surplus_kw") for n in ns], **kw)
    for ax, title, ylabel in zip(
        axes,
        ["How often each method finds the optimal plan", "Energy wasted or held in reserve"],
        ["% of instances solved optimally", "mean surplus supply (kW)"],
    ):
        ax.set_xticks(np.arange(len(ns)))
        ax.set_xticklabels([f"n = {n}" for n in ns])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(0, 105)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=len(present), bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    paths.append(out / "baselines.png")
    fig.savefig(paths[-1], dpi=170)
    plt.close(fig)
    return paths


INT_FIELDS = {"n", "edges", "couplings", "seed", "p", "is_optimal", "nfev", "depth", "two_qubit_gates",
              "two_qubit_depth"}


def read_csv(path: Path) -> list[dict]:
    """Load benchmark.csv back into rows with numeric types (for --replot)."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for k, v in raw.items():
                if k == "method" or v == "":
                    row[k] = v
                elif k in INT_FIELDS:
                    row[k] = int(float(v))
                else:
                    row[k] = float(v)
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--density", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--objective", choices=["expectation", "cvar"], default="expectation")
    parser.add_argument("--alpha", type=float, default=0.25, help="CVaR tail fraction")
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--transpile", action="store_true", help="report transpiled depth on the fake backend")
    parser.add_argument("--noisy", action="store_true", help="also run final parameters on a noisy Heron model")
    parser.add_argument("--noisy-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--backend", default="FakeFez")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--replot", action="store_true",
                        help="only rebuild summary.md and figures from an existing benchmark.csv")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    if args.noisy:
        args.transpile = True

    if args.replot:
        rows = read_csv(args.out / "benchmark.csv")
        args.max_p = max(r["p"] for r in rows)
    else:
        rows = run(args)
        write_csv(rows, args.out / "benchmark.csv")
    summary = summarize(rows, args.max_p)
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    figures = plot(rows, args.out / "figures", args.max_p)
    print(summary)
    print("figures:", ", ".join(str(p) for p in figures))


if __name__ == "__main__":
    main()
