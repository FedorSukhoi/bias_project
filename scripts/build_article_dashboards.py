"""Build three editorial dashboards used by the project README and article draft."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/publication_proxy_headlines.csv"
SPLIT_PATH = ROOT / "data/splits/pilot_v1.csv"
COMPARISON_PATH = ROOT / "reports/modeling/development_model_comparison.csv"
OOD_METRICS_PATH = ROOT / "reports/modeling/ood_metrics.json"
OOD_PREDICTIONS_PATH = ROOT / "reports/modeling/ood_predictions.csv"
OUTPUT_DIR = ROOT / "reports/figures"

LABELS = ["Left", "Lean Left", "Center", "Lean Right", "Right"]
LABEL_COLORS = {
    "Left": "#315B7D",
    "Lean Left": "#6E9FBF",
    "Center": "#858585",
    "Lean Right": "#D28A67",
    "Right": "#AD4D43",
}
BG = "#F7F4EE"
INK = "#17212B"
MUTED = "#5E6872"
LINE = "#D8D3C9"
ACCENT = "#C7983D"
GOOD = "#417A66"
BAD = "#A84F48"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": LINE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
        }
    )


def add_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.955, title, fontsize=25, fontweight="bold", color=INK, va="top")
    fig.text(0.055, 0.905, subtitle, fontsize=12.5, color=MUTED, va="top")


def rounded_box(ax, xy, width, height, facecolor="#FFFFFF", edgecolor=LINE, radius=0.025):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1,
        facecolor=facecolor,
        edgecolor=edgecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def build_overview(data: pd.DataFrame, manifest: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(15, 8.5), facecolor=BG)
    add_title(
        fig,
        "The experiment in one view",
        "Can a transparent word-based model infer a publisher's political-bias category from one headline?",
    )

    stat_ax = fig.add_axes([0.055, 0.72, 0.89, 0.14])
    stat_ax.axis("off")
    stats = [
        (f"{len(data):,}", "political headlines"),
        (f"{data['source_id'].nunique()}", "publications"),
        ("5", "weak outlet labels"),
        (f"{manifest.loc[manifest['split'].eq('test_ood'), 'source_id'].nunique()}", "unseen test publishers"),
    ]
    for index, (value, label) in enumerate(stats):
        x = index * 0.25
        rounded_box(stat_ax, (x + 0.01, 0.02), 0.225, 0.88, facecolor="#FFFEFA")
        stat_ax.text(x + 0.035, 0.60, value, transform=stat_ax.transAxes, fontsize=24, fontweight="bold", color=INK)
        stat_ax.text(x + 0.035, 0.27, label, transform=stat_ax.transAxes, fontsize=11.5, color=MUTED)

    class_ax = fig.add_axes([0.075, 0.24, 0.40, 0.40])
    counts = data["weak_label"].value_counts().reindex(LABELS)
    y = np.arange(len(LABELS))
    class_ax.barh(y, counts.values, color=[LABEL_COLORS[label] for label in LABELS], height=0.58)
    class_ax.set_yticks(y, LABELS)
    class_ax.invert_yaxis()
    class_ax.set_xlim(0, max(counts) * 1.23)
    class_ax.set_title("The archive is not class-balanced", loc="left", fontsize=15, fontweight="bold", pad=14)
    class_ax.set_xlabel("Headlines")
    class_ax.spines[["top", "right", "left"]].set_visible(False)
    class_ax.grid(axis="x", color=LINE, linewidth=0.8, alpha=0.8)
    class_ax.set_axisbelow(True)
    for yi, value in zip(y, counts.values):
        class_ax.text(value + 7, yi, f"{value:,}", va="center", color=INK, fontweight="bold")

    flow_ax = fig.add_axes([0.53, 0.25, 0.415, 0.39])
    flow_ax.axis("off")
    flow_ax.text(0, 1.02, "A deliberately difficult test", transform=flow_ax.transAxes,
                 fontsize=15, fontweight="bold", color=INK)
    steps = [
        (0.69, "1", "Read only the headline", "No outlet name, URL, author or article body"),
        (0.41, "2", "Learn from 12 publishers", "Older headlines train; newer ones validate"),
        (0.13, "3", "Switch the publishers", "Five entirely unseen outlets form the final test"),
    ]
    for y, number, heading, note in steps:
        rounded_box(flow_ax, (0.03, y), 0.92, 0.19, facecolor="#FFFEFA")
        flow_ax.text(0.085, y + 0.095, number, transform=flow_ax.transAxes, fontsize=11,
                     color="#FFFFFF", fontweight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="circle,pad=0.35", facecolor=LABEL_COLORS["Center"], edgecolor="none"))
        flow_ax.text(0.16, y + 0.128, heading, transform=flow_ax.transAxes,
                     fontsize=12.2, fontweight="bold", color=INK, va="center")
        flow_ax.text(0.16, y + 0.057, note, transform=flow_ax.transAxes,
                     fontsize=10.2, color=MUTED, va="center")
    flow_ax.annotate("", xy=(0.085, 0.62), xytext=(0.085, 0.67), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    flow_ax.annotate("", xy=(0.085, 0.34), xytext=(0.085, 0.39), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))

    note_ax = fig.add_axes([0.055, 0.075, 0.89, 0.10])
    note_ax.axis("off")
    rounded_box(note_ax, (0, 0.03), 1, 0.88, facecolor="#EFE9DD", edgecolor="#D5C7AD")
    note_ax.text(0.025, 0.58, "The essential caveat", transform=note_ax.transAxes,
                 fontsize=11.5, fontweight="bold", color=INK, va="center")
    note_ax.text(
        0.19,
        0.58,
        "Each label belongs to the publication—not to a human judgment of that individual headline.",
        transform=note_ax.transAxes,
        fontsize=12.2,
        color=INK,
        va="center",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "01_experiment_overview.png", dpi=190, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def build_result_dashboard(
    comparison: pd.DataFrame,
    ood_metrics: dict[str, float],
    predictions: pd.DataFrame,
) -> None:
    validation = comparison.loc["word_unigrams"]
    test_macro = ood_metrics["macro_f1"]
    relative_drop = 1 - test_macro / validation["macro_f1"]

    fig = plt.figure(figsize=(15, 8.5), facecolor=BG)
    add_title(
        fig,
        "The result changed when the publishers changed",
        "Performance on newer headlines from familiar publishers did not carry over to unseen publishers.",
    )

    score_ax = fig.add_axes([0.07, 0.22, 0.38, 0.60])
    names = ["Familiar\npublishers", "Unseen\npublishers"]
    values = [validation["macro_f1"], test_macro]
    bars = score_ax.bar(names, values, color=[GOOD, BAD], width=0.58)
    score_ax.set_ylim(0, 0.52)
    score_ax.set_ylabel("Macro-F1 (higher is better)")
    score_ax.set_title("The primary score more than halved", loc="left", fontsize=15, fontweight="bold", pad=16)
    score_ax.spines[["top", "right"]].set_visible(False)
    score_ax.grid(axis="y", color=LINE, linewidth=0.8)
    score_ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        score_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}",
                      ha="center", fontsize=17, fontweight="bold", color=INK)
    score_ax.text(
        0.5,
        0.48,
        f"{relative_drop:.0%} lower",
        transform=score_ax.transAxes,
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=BAD,
    )
    score_ax.text(
        0.02,
        -0.20,
        "Macro-F1 gives each of the five categories equal importance.",
        transform=score_ax.transAxes,
        fontsize=10.5,
        color=MUTED,
    )

    matrix_ax = fig.add_axes([0.54, 0.27, 0.39, 0.47])
    matrix = confusion_matrix(
        predictions["weak_label"], predictions["prediction"], labels=LABELS, normalize="true"
    )
    image = matrix_ax.imshow(matrix, cmap="YlOrBr", vmin=0, vmax=1)
    matrix_ax.set_xticks(range(5), LABELS, rotation=28, ha="right")
    matrix_ax.set_yticks(range(5), LABELS)
    matrix_ax.set_xlabel("Model prediction")
    matrix_ax.set_ylabel("Publisher's weak label")
    matrix_ax.set_title("Where unseen-outlet predictions landed", loc="left", fontsize=15, fontweight="bold", pad=16)
    for row in range(5):
        for col in range(5):
            value = matrix[row, col]
            matrix_ax.text(col, row, f"{value:.0%}", ha="center", va="center",
                           color="#FFFFFF" if value > 0.55 else INK, fontsize=10.5,
                           fontweight="bold" if row == col else "normal")
    for spine in matrix_ax.spines.values():
        spine.set_edgecolor(LINE)
    callout_ax = fig.add_axes([0.50, 0.075, 0.445, 0.095])
    callout_ax.axis("off")
    rounded_box(callout_ax, (0, 0.04), 1, 0.88, facecolor="#EFE9DD", edgecolor="#D5C7AD")
    callout_ax.text(
        0.035,
        0.54,
        fill("The model recognized familiar publishing ecosystems better than it recognized a transferable idea of political bias.", 75),
        transform=callout_ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        va="center",
    )

    fig.savefig(OUTPUT_DIR / "02_generalization_gap.png", dpi=190, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def build_complexity_dashboard() -> None:
    fig = plt.figure(figsize=(15, 8.5), facecolor=BG)
    add_title(
        fig,
        "Why political bias is harder than a vocabulary list",
        "A headline model sees real framing signals—but those signals arrive mixed with topics, habits and missing context.",
    )

    ax = fig.add_axes([0.055, 0.14, 0.89, 0.70])
    ax.axis("off")
    columns = [
        (
            0.00,
            "What one headline can reveal",
            GOOD,
            [
                ("Word choice", "Loaded or emotionally intensified language"),
                ("Framing", "Who is praised, blamed or quoted"),
                ("Style", "Punctuation, certainty and recurring phrasing"),
                ("Story focus", "Which actor, policy or conflict is foregrounded"),
            ],
        ),
        (
            0.345,
            "What the model may confuse with bias",
            ACCENT,
            [
                ("Publisher habits", "House style and repeated headline templates"),
                ("Recurring names", "Authors, programs, politicians and institutions"),
                ("Topic mix", "Different outlets selecting different stories"),
                ("Moment in time", "Events concentrated in uneven archive periods"),
            ],
        ),
        (
            0.69,
            "What the headline cannot reveal",
            BAD,
            [
                ("Full context", "Balance, evidence and qualifications in the article"),
                ("Omitted coverage", "Stories the publication chose not to run"),
                ("Placement", "Homepage prominence, imagery and repetition"),
                ("Reader judgment", "Whether people across viewpoints perceive bias"),
            ],
        ),
    ]
    for x, heading, color, items in columns:
        rounded_box(ax, (x, 0.13), 0.305, 0.76, facecolor="#FFFEFA")
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.82), 0.305, 0.07,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                transform=ax.transAxes, facecolor=color, edgecolor=color,
            )
        )
        ax.text(x + 0.018, 0.855, fill(heading, 36), transform=ax.transAxes,
                color="#FFFFFF" if color != ACCENT else INK, fontsize=12.5,
                fontweight="bold", va="center")
        for index, (label, note) in enumerate(items):
            y = 0.72 - index * 0.145
            ax.text(x + 0.025, y, label, transform=ax.transAxes, fontsize=11.8,
                    fontweight="bold", color=INK, va="top")
            ax.text(x + 0.025, y - 0.045, fill(note, 38), transform=ax.transAxes,
                    fontsize=10.3, color=MUTED, va="top", linespacing=1.28)

    ax.text(0.0, 0.04, "Examples the first model associated with publication classes", transform=ax.transAxes,
            fontsize=12.5, fontweight="bold", color=INK)
    examples = [
        ("Left", "american · china · ukraine"),
        ("Lean Left", "senate · republican · texas"),
        ("Center", "department · tariffs · trade"),
        ("Lean Right", "circuit · court · brickbat"),
        ("Right", "dem · nolte · illegal"),
    ]
    for index, (label, words) in enumerate(examples):
        x = index * 0.20
        ax.text(x, -0.015, label, transform=ax.transAxes, color=LABEL_COLORS[label],
                fontsize=10.5, fontweight="bold")
        ax.text(x, -0.055, words, transform=ax.transAxes, color=MUTED, fontsize=9.6)

    note_ax = fig.add_axes([0.055, 0.035, 0.89, 0.06])
    note_ax.axis("off")
    note_ax.text(
        0,
        0.5,
        "These are sample associations in this dataset—not definitions of Left, Center or Right, and not proof that a word is inherently biased.",
        fontsize=10.8,
        color=MUTED,
        style="italic",
        va="center",
    )

    fig.savefig(OUTPUT_DIR / "03_why_bias_is_complex.png", dpi=190, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    setup_style()
    data = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(SPLIT_PATH)
    comparison = pd.read_csv(COMPARISON_PATH, index_col=0)
    ood_metrics = json.loads(OOD_METRICS_PATH.read_text())
    predictions = pd.read_csv(OOD_PREDICTIONS_PATH)

    build_overview(data, manifest)
    build_result_dashboard(comparison, ood_metrics, predictions)
    build_complexity_dashboard()
    print(f"Wrote three dashboards to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
