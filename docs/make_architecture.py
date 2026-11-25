"""Render docs/architecture.png, a one-slide architecture diagram for the pitch."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1F2A37"
MUTED = "#5B6675"
GREEN = "#2E9E5B"
BLUE = "#2F6FB0"
AMBER = "#C98400"

boxes = {
    # key: (x, y, w, h, title, subtitle, color)
    "grid": (0.2, 3.1, 2.3, 1.3, "Microgrid", "loads, solar, wind,\nbatteries, diesel", GREEN),
    "qubo": (3.0, 3.1, 2.3, 1.3, "Dispatch QUBO", "cost + local balance\n+ 15% reserve", GREEN),
    "ising": (5.8, 3.1, 2.3, 1.3, "Ising Hamiltonian", "SparsePauliOp\nx = (1 - Z) / 2", BLUE),
    "qaoa": (8.6, 3.1, 2.3, 1.3, "QAOA circuit", "RZZ cost layer, RX mixer\nINTERP warm start", BLUE),
    "ideal": (8.6, 1.35, 2.3, 1.1, "Ideal simulator", "optimise parameters", BLUE),
    "noisy": (5.8, 1.35, 2.3, 1.1, "Noisy Heron model", "AerSimulator + SamplerV2", BLUE),
    "hw": (3.0, 1.35, 2.3, 1.1, "IBM Heron QPU", "SamplerV2 DD + twirling\nEstimatorV2 ZNE", BLUE),
    "base": (0.2, 1.35, 2.3, 1.1, "Classical baselines", "exact, greedy, SA", AMBER),
    "out": (3.0, -0.3, 5.1, 1.0, "Dispatch decision + metrics", "% demand met, surplus, diesel kW, cost; live demo app", GREEN),
}

fig, ax = plt.subplots(figsize=(11.5, 5.6))
ax.set_xlim(0, 11.1)
ax.set_ylim(-0.5, 4.6)
ax.axis("off")

for x, y, w, h, title, sub, color in boxes.values():
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="white", edgecolor=color, linewidth=2))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=11.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=8.8, color=MUTED, linespacing=1.3)


def arrow(a, b, start="right", end="left"):
    def anchor(key, side):
        x, y, w, h = boxes[key][:4]
        return {"right": (x + w, y + h / 2), "left": (x, y + h / 2),
                "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}[side]

    ax.add_patch(FancyArrowPatch(anchor(a, start), anchor(b, end), arrowstyle="-|>", mutation_scale=14,
                                 color=MUTED, linewidth=1.4, shrinkA=2, shrinkB=2))


arrow("grid", "qubo")
arrow("qubo", "ising")
arrow("ising", "qaoa")
arrow("qaoa", "ideal", "bottom", "top")
arrow("ideal", "noisy", "left", "right")
arrow("noisy", "hw", "left", "right")
arrow("qubo", "base", "bottom", "top")
arrow("hw", "out", "bottom", "top")
arrow("base", "out", "bottom", "left")
ax.text(9.75, 0.95, "same parameters reused:\nideal -> noisy -> hardware", ha="center", fontsize=8.5, color=MUTED,
        style="italic")

out = Path(__file__).with_name("architecture.png")
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", out.name)
