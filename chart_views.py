"""Readable, self-contained Cell-style figures for the application UI."""

from __future__ import annotations

import base64
import io
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": ["Arial", "sans-serif"],
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CELL_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]


def canvas(title, wide=False):
    # Compact UI renditions keep publication typography without empty gutters.
    fig, ax = plt.subplots(figsize=(7.0, 2.8) if wide else (3.5, 3.0))
    ax.set_title(title, loc="left", pad=10)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    ax.tick_params(direction="out")
    ax.grid(False)
    return fig, ax


def finish(fig, caption):
    """Reserve a real caption region inside the image, outside the data axes."""
    wide = fig.get_figwidth() > 5
    caption_lines = textwrap.fill(caption, width=112 if wide else 53)
    lines = len(caption_lines.splitlines())
    footer = (lines * 10.4 + 7) / (72 * fig.get_figheight())
    fig.tight_layout(pad=0.65, rect=(0, footer, 1, 1))
    fig.text(0.04, 0.025, caption_lines, ha="left", va="bottom",
             fontsize=8, linespacing=1.3, color=CELL_COLORS[3])
    return fig


def png(fig):
    buffer = io.BytesIO()
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(buffer, format="png", dpi=300, bbox_inches=None)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def legend(ax, **kwargs):
    return ax.legend(frameon=True, facecolor="white", framealpha=0.94,
                     edgecolor="none", fontsize=8, **kwargs)


def cindex_figure(data):
    fig, ax = canvas("Concordance | C-index")
    for index, (name, result, train) in enumerate((
        ("Cox PH", data["RES_COX"], data["TR_COX"]),
        ("AFT", data["RES_AFT"], data["TR_AFT"]),
    )):
        color = CELL_COLORS[index]
        score = result["c_index"]
        ax.errorbar(score, 1 - index, xerr=[[score - result["ci_lo"]], [result["ci_hi"] - score]],
                    fmt="o", color=color, markersize=4, elinewidth=0.8, capsize=3)
        ax.plot(train["c_index"], 1 - index + 0.18, marker="o", markersize=4,
                markerfacecolor="white", markeredgecolor=color, linestyle="none")
        ax.text(0.98, 1 - index + 0.28, f"{name}: {score:.3f}",
                transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color=color)
    ax.axvline(0.5, color=CELL_COLORS[2], linestyle=":", linewidth=1)
    ax.set(xlim=(0.45, 0.95), ylim=(-0.4, 1.65), xlabel="C-index",
           yticks=[0, 1], yticklabels=["AFT", "Cox PH"])
    ax.plot([], [], "o", color=CELL_COLORS[3], markersize=4, label="Test (95% CI)")
    ax.plot([], [], "o", markerfacecolor="white", markeredgecolor=CELL_COLORS[3],
            markersize=4, label="Train (apparent)")
    legend(ax, loc="upper left", bbox_to_anchor=(0, 1), ncol=1)
    return finish(fig, f"TCGA-LUAD | Train n={data['N_TRAIN']}; test n={data['N_TEST']}. "
                  "Test CI: 150 bootstraps.")


def auc_figure(data):
    fig, ax = canvas("Time-dependent discrimination")
    for index, (name, result) in enumerate((("Cox PH", data["RES_COX"]), ("AFT", data["RES_AFT"]))):
        ax.plot(result["times"], result["auc_vals"], color=CELL_COLORS[index],
                linewidth=1, linestyle="-" if index == 0 else "--",
                marker="o", markersize=4, label=f"{name} mean = {result['mean_auc']:.3f}")
    ax.axhline(0.5, color=CELL_COLORS[2], linestyle=":", linewidth=1)
    ax.set(ylim=(0.45, 1), xlabel="Time (months)", ylabel="Cumulative / dynamic AUC",
           xticks=data["RES_COX"]["times"])
    legend(ax, loc="upper right")
    return finish(fig, f"TCGA-LUAD test n={data['N_TEST']} | "
                  "Training-based IPCW; dotted line: chance.")


def distribution_figure(data):
    fig, ax = canvas("Marginal survival fits")
    times, survival = data["_survival_function_frame"](data["KM_TRAIN"])
    limit = min(240.0, float(np.nanmax(times)))
    grid = np.linspace(0, limit, 260)
    ax.step(times, survival, where="post", color=CELL_COLORS[0],
            linewidth=1, label="Kaplan-Meier")
    for index, (name, fitter) in enumerate(data["DIST"]["fitters"].items(), start=1):
        best = name == data["DIST"]["best"]
        ax.plot(grid, data["_survival_at_times"](fitter, grid),
                color=CELL_COLORS[index], linewidth=1, linestyle="-" if best else "--",
                label=name + (" (best)" if best else ""))
    ax.set(xlim=(0, limit), ylim=(0, 1.05), xlabel="Time (months)", ylabel="Survival probability")
    legend(ax, loc="upper right", handlelength=1.6, labelspacing=0.35)
    return finish(fig, f"TCGA-LUAD | Training n={data['N_TRAIN']}. "
                  "Censoring-aware marginal fits; best: lowest AIC.")


def survival_figure(times, cox, aft, narrow=False):
    fig, ax = canvas("Patient survival projections", wide=not narrow)
    ax.plot(times, cox, color=CELL_COLORS[0], linewidth=1, label="Cox PH")
    ax.plot(times, aft, color=CELL_COLORS[1], linewidth=1, linestyle="--",
            label="Log-Logistic AFT")
    ax.fill_between(times, cox, aft, color=CELL_COLORS[2], alpha=0.14, linewidth=0)
    ax.set(xlim=(0, 72), ylim=(0, 1.04), xticks=[0, 12, 24, 36, 48, 60, 72],
           yticks=[0, 0.25, 0.5, 0.75, 1], xlabel="Time (months)", ylabel="Survival probability")
    legend(ax, loc="lower left")
    return finish(fig, "TCGA-LUAD | Shading: model disagreement, not a confidence interval. Research only.")
