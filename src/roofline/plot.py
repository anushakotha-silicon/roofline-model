"""Optional matplotlib rendering of a roofline chart."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_roofline(hw, results, path: str) -> None:
    ai = np.logspace(-1, 4, 200)
    roof = np.minimum(hw.peak_flops, ai * hw.mem_bw) / 1e12

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(ai, roof, color="#333", lw=2, label=hw.name)
    ax.axvline(hw.ridge_point, color="#999", ls="--", lw=1)
    for r in results:
        ax.scatter(r.kernel.intensity, r.attainable_flops / 1e12, zorder=3)
        ax.annotate(r.kernel.name, (r.kernel.intensity, r.attainable_flops / 1e12),
                    textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Attainable TFLOP/s")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
