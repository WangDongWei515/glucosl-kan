"""Render the approved Figure 1 from existing learned curves and cohort records."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, PathPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[2]
OUT = STAGE / "paper/figures"
COHORT_GREY = "#85919b"
BLUE, RED, TEAL = "#1f4e79", "#c0504d", "#2a9d8f"
DARK, GREY, LIGHT = "#25333e", "#7b858d", "#cbd2d8"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "text.color": DARK,
        "axes.labelcolor": DARK,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def cohort_text(ax, x, y, s, **kwargs):
    return ax.text(x, y, s, transform=ax.transAxes, **kwargs)

def cohort_arrow(ax, start, end, color=COHORT_GREY, bend=0.5):
    x0, y0 = start
    x1, y1 = end
    dx = (x1 - x0) * bend
    path = MplPath([(x0, y0), (x0 + dx, y0), (x1 - dx, y1), (x1, y1)],
                   [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=10,
                                lw=1.25, color=color, transform=ax.transAxes))

def draw_nested_cohort(ax):
    cohort_text(ax, 0.035, 0.94, "CGMacros (US)", fontsize=13, weight="bold", va="center")
    ax.add_patch(Ellipse((0.255, 0.465), 0.440, 0.745, facecolor="#fafbfc",
                         edgecolor="#a9b2ba", lw=1.2, transform=ax.transAxes))
    cohort_text(ax, 0.255, 0.735, "ALL PARTICIPANTS", fontsize=8.1, color=COHORT_GREY,
        ha="center", va="center")
    cohort_text(ax, 0.255, 0.665, "44", fontsize=14.0, weight="bold",
        ha="center", va="center")
    cohort_text(ax, 0.255, 0.590, "participants", fontsize=9.8,
        ha="center", va="center")
    cohort_text(ax, 0.255, 0.535, "1,147 meals", fontsize=9.2, color=COHORT_GREY,
        ha="center", va="center")

    ax.add_patch(Ellipse((0.255, 0.310), 0.335, 0.350, facecolor="#f9eaea",
                         edgecolor=RED, lw=1.35, transform=ax.transAxes))
    cohort_text(ax, 0.255, 0.407, "14", fontsize=13.5, color=RED,
        weight="bold", ha="center", va="center")
    cohort_text(ax, 0.255, 0.345, "participants", fontsize=9.3, color=RED,
        ha="center", va="center")
    cohort_text(ax, 0.255, 0.283, "HbA1c ≥6.5%", fontsize=9.1, color=RED,
        ha="center", va="center")
    cohort_text(ax, 0.255, 0.218, "353 meals", fontsize=9.0, color=RED,
        ha="center", va="center")

    cohort_arrow(ax, (0.454, 0.625), (0.625, 0.735))
    cohort_arrow(ax, (0.472, 0.530), (0.625, 0.520))
    cohort_arrow(ax, (0.423, 0.310), (0.625, 0.245), color=RED)
    cohort_text(ax, 0.655, 0.735, "Supplementary\nvalidation", fontsize=11.0,
        va="center", linespacing=1.24)
    cohort_text(ax, 0.655, 0.520, "US model\ntraining", fontsize=11.0,
        va="center", linespacing=1.24)
    cohort_text(ax, 0.655, 0.245, "Cross-country\nvalidation", fontsize=11.0,
        color=RED, va="center", linespacing=1.24)


def build(output_dir=OUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(STAGE / "results/glucosl/interpretability/partial_dependence.csv")
    curves = curves.loc[curves.model == "Shanghai-trained"]
    split = pd.read_csv(STAGE / "data/processed/glucosl/split.csv", parse_dates=["first_recording"])
    meals = pd.read_csv(STAGE / "data/processed/cgmacros/primary_cohort.csv")
    subjects = meals.groupby("subject").hba1c_pct.first().sort_values(ascending=False)
    assert split.groupby("set").size().to_dict() == {"A": 67, "B": 33}
    assert split.groupby("set").meals.sum().to_dict() == {"A": 1952, "B": 670}
    assert len(subjects) == 44 and int((subjects >= 6.5).sum()) == 14
    assert len(meals) == 1147 and int((meals.hba1c_pct >= 6.5).sum()) == 353

    width, height = 10.6, 7.6
    fig = plt.figure(figsize=(width, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def text(x, y, label, **kwargs):
        return ax.text(x, y, label, **kwargs)

    def line(x0, y0, x1, y1, color, lw=1.1, alpha=0.67):
        ax.add_line(Line2D([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=alpha))

    def bend(x0, y0, x1, y1, color, alpha=0.62):
        mid = (x0 + x1) / 2
        path = MplPath(
            [(x0, y0), (mid, y0), (mid, y1), (x1, y1)],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=1.1, alpha=alpha))

    def node(x, y, symbol, color, radius=0.0155):
        ax.add_patch(
            Ellipse((x, y), radius * 2, radius * 2 * width / height,
                    facecolor="white", edgecolor=color, linewidth=1.45, zorder=4)
        )
        text(x, y, symbol, color=color, ha="center", va="center", fontsize=12, zorder=5)

    text(0.025, 0.969, "a", fontweight="bold", fontsize=15, va="center")
    text(0.060, 0.969, "GlucoSL-KAN architecture", fontsize=14, fontweight="bold", va="center")
    text(0.060, 0.935, "Single-member structure; regression and classification are fitted separately",
         fontsize=9, color=GREY, va="center")

    curve_x, curve_w = 0.210, 0.128
    sum_x, factor_x, product_x = 0.445, 0.565, 0.710
    curve_records = []
    branches = [
        ("sensitivity", BLUE, ["age", "sex", "bmi", "hba1c_pct", "fasting_glucose_mgdl"],
         ["Age", "Sex", "BMI", "HbA1c", "Fasting glucose"],
         np.linspace(0.882, 0.696, 5), "s", "Personal sensitivity", (0.55, 1.45)),
        ("load", RED, ["carbohydrate_g", "protein_g", "fat_g", "fiber_g", "hour"],
         ["Carbohydrate", "Protein", "Fat", "Fiber", "Clock time"],
         np.linspace(0.599, 0.413, 5), "l", "Meal load", (15, 80)),
    ]
    for part, color, inputs, labels, ys, symbol, name, ylim in branches:
        center = float(np.mean(ys))
        for column, label, y in zip(inputs, labels, ys):
            text(0.190, y, label, ha="right", va="center", fontsize=10.1)
            rows = curves.loc[(curves.part == part) & (curves.input == column)].sort_values("value")
            assert len(rows)
            curve_records.append({"part": part, "input": column, "rows": len(rows)})
            inset = fig.add_axes([curve_x, y - 0.0185, curve_w, 0.037])
            if column == "sex":
                inset.plot(rows.value, rows.estimate, "o", markersize=4, color=color)
                inset.set_xlim(-0.5, 1.5)
            else:
                inset.plot(rows.value, rows.estimate, color=color, linewidth=1.65)
                inset.fill_between(rows.value, rows.low, rows.high, color=color, alpha=0.12, linewidth=0)
                inset.set_xlim(rows.value.min(), rows.value.max())
            inset.set_ylim(*ylim)
            inset.set_xticks([])
            inset.set_yticks([])
            if part == "sensitivity":
                inset.axhline(1, color=LIGHT, linestyle=":", linewidth=0.7, zorder=0)
            for side in ["top", "right", "left"]:
                inset.spines[side].set_visible(False)
            inset.spines["bottom"].set_color(LIGHT)
            inset.spines["bottom"].set_linewidth(0.65)
            if column in {"carbohydrate_g", "fiber_g"}:
                inset.text(0.98, 0.88, "↑" if column == "carbohydrate_g" else "↓",
                           transform=inset.transAxes, color=color, ha="right", va="top", fontsize=10)
            bend(curve_x + curve_w + 0.006, float(y), sum_x - 0.016, center, color)
        node(sum_x, center, "Σ", color)
        line(sum_x + 0.016, center, factor_x - 0.016, center, color)
        text((sum_x + factor_x) / 2, center + 0.020, "softplus", color=GREY,
             ha="center", va="bottom", fontsize=8.2)
        node(factor_x, center, r"$s$" if symbol == "s" else r"$\ell$", color)
        text(factor_x, center - 0.035, "Personal\nsensitivity" if part == "sensitivity" else name,
             ha="center", va="top", fontsize=9.8, color=color, linespacing=1.12)
        bend(factor_x + 0.016, center, product_x - 0.016, 0.648, color)

    node(product_x, 0.648, "×", DARK)
    bend(product_x + 0.016, 0.648, 0.785, 0.721, DARK, alpha=0.45)
    bend(product_x + 0.016, 0.648, 0.785, 0.574, DARK, alpha=0.45)
    for y, title, label in [(0.721, "Regression", "2-h iAUC"),
                            (0.574, "Classification", "P(peak ≥180 mg/dL)")]:
        text(0.798, y + 0.018, title, fontsize=10.6, fontweight="bold", va="center")
        text(0.798, y - 0.013, label, fontsize=10.2, va="center")

    text(0.210, 0.365, "Final prediction: mean of 5 independently trained members",
         fontsize=9.5, color=GREY, va="center")
    line(0.025, 0.339, 0.975, 0.339, LIGHT, lw=0.55, alpha=0.8)

    text(0.025, 0.309, "b", fontweight="bold", fontsize=15, va="center")
    text(0.060, 0.309, "Study design", fontsize=14, fontweight="bold", va="center")
    text(0.200, 0.265, "ShanghaiT2DM", fontsize=11.2, fontweight="bold", va="center")

    timeline = fig.add_axes([0.200, 0.070, 0.410, 0.170])
    offsets = {"A": 0.48, "B": -0.25}
    for set_, color in [("A", BLUE), ("B", TEAL)]:
        dates = split.loc[split.set == set_, "first_recording"]
        jitter = np.random.default_rng(0).uniform(-0.09, 0.09, len(dates))
        timeline.scatter(dates, offsets[set_] + jitter, s=16, color=color, alpha=0.83, linewidths=0)
    cut = pd.Timestamp("2021-06-29")
    timeline.axvline(cut, color=GREY, linestyle="--", linewidth=0.85)
    timeline.text(cut + pd.Timedelta(days=5), 0.91, "29 Jun 2021", fontsize=8.2, color=GREY, va="top")
    timeline.set(xlim=(pd.Timestamp("2020-08-10"), pd.Timestamp("2022-02-15")), ylim=(-0.66, 0.98))
    timeline.set_yticks([])
    timeline.set_xticks(pd.to_datetime(["2020-09-01", "2021-01-01", "2021-05-01", "2021-09-01", "2022-01-01"]))
    timeline.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    timeline.tick_params(axis="x", labelsize=8.5, length=3, color=GREY, pad=5)
    timeline.spines["left"].set_visible(False)
    timeline.spines["bottom"].set_color(LIGHT)
    timeline.spines["bottom"].set_linewidth(0.75)
    timeline.set_xlabel("First recording date", fontsize=9.3, labelpad=7)
    for set_, color, title, stats in [("A", BLUE, "A  Development", "67 patients / 1,952 meals"),
                                      ("B", TEAL, "B  Validation", "33 patients / 670 meals")]:
        y = 0.070 + (offsets[set_] + 0.66) / (0.98 + 0.66) * 0.170
        text(0.187, y + 0.009, title, fontsize=9.2, color=color, ha="right", va="center", fontweight="bold")
        text(0.187, y - 0.015, stats, fontsize=8.5, color=color, ha="right", va="center")

    # Selected option A: nested cohort plus curved links to the data uses.
    # Keep the model and Shanghai timeline at their existing coordinates.
    cohort = fig.add_axes([0.665, 0.015, 0.315, 0.266])
    cohort.set(xlim=(0, 1), ylim=(0, 1))
    cohort.axis("off")
    draw_nested_cohort(cohort)
    for label in cohort.texts:
        if label.get_text() == "CGMacros (US)":
            label.set_fontsize(11.2)
        else:
            label.set_fontsize(max(6.4, label.get_fontsize() * 0.72))
    total = len(subjects)
    selected = int((subjects >= 6.5).sum())

    fig.savefig(output_dir / "Figure_1.pdf", facecolor="white")
    fig.savefig(output_dir / "Figure_1.png", dpi=240, facecolor="white")
    plt.close(fig)
    return {"pdf": str(output_dir / "Figure_1.pdf"), "png": str(output_dir / "Figure_1.png")}


if __name__ == "__main__":
    build()
