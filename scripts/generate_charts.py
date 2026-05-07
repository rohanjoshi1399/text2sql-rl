"""Generate presentation charts from evaluation results."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

# Load data. Prefer the *_official_hardness.json variants (produced by
# scripts/reclassify_difficulty.py) so per-difficulty charts use Spider's
# canonical hardness partition instead of our in-repo heuristic.
def _load(name: str) -> dict:
    official = f"results/eval_dev_{name}_official_hardness.json"
    default = f"results/eval_dev_{name}.json"
    path = official if os.path.exists(official) else default
    d = json.load(open(path))
    # Normalize per-difficulty field name so existing chart code keeps working.
    if "by_difficulty_official" in d:
        d["by_difficulty"] = d["by_difficulty_official"]
    return d

zs = _load("zero-shot")
fs = _load("few-shot")
sft = _load("model")
grpo = _load("grpo_best") if os.path.exists(
    "results/eval_dev_grpo_best_official_hardness.json"
) or os.path.exists("results/eval_dev_grpo_best.json") else None

# ============================================================
# Chart 1: Overall Results Bar Chart
# ============================================================
conditions = ["Zero-shot", "5-shot", "DSPy\n(MIPROv2)", "SFT\n(ep 1)", "SFT\n(ep 3)"]
scores = [67.8, 66.3, 58.6, 69.2, 71.1]
colors = ["#4A90D9", "#5BA0E0", "#E8A838", "#58D68D", "#27AE60"]

if grpo is not None:
    conditions.append("GRPO\n(final)")
    scores.append(round(grpo["overall_ex"] * 100, 1))
    colors.append("#D35400")

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(conditions, scores, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
            f"{score:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=14)

ax.axhline(y=29.2, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(len(conditions) - 0.6, 30.5, "Update 1: Exact Match (29.2%)",
        color="red", fontsize=10, alpha=0.8, ha="right")

ax.set_ylabel("Execution Accuracy (%)", fontsize=13)
ax.set_title("Spider Dev Set - Execution Accuracy (n=1,034)", fontsize=15, fontweight="bold")
ax.set_ylim(0, 85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/overall_results.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/overall_results.png")

# ============================================================
# Chart 2: EX by Difficulty - Grouped Bar Chart
# ============================================================
difficulties = ["Easy", "Medium", "Hard", "Extra Hard"]

zs_scores = [zs["by_difficulty"].get("easy", 0) * 100, zs["by_difficulty"].get("medium", 0) * 100,
             zs["by_difficulty"].get("hard", 0) * 100, zs["by_difficulty"].get("extra", 0) * 100]
fs_scores = [fs["by_difficulty"].get("easy", 0) * 100, fs["by_difficulty"].get("medium", 0) * 100,
             fs["by_difficulty"].get("hard", 0) * 100, fs["by_difficulty"].get("extra", 0) * 100]
sft_scores = [sft["by_difficulty"].get("easy", 0) * 100, sft["by_difficulty"].get("medium", 0) * 100,
              sft["by_difficulty"].get("hard", 0) * 100, sft["by_difficulty"].get("extra", 0) * 100]

x = np.arange(len(difficulties))

# Include GRPO as a fourth series when its eval JSON is available.
series = [("Zero-shot", zs_scores, "#4A90D9"),
          ("5-shot", fs_scores, "#5BA0E0"),
          ("SFT (3 epochs)", sft_scores, "#27AE60")]
if grpo is not None:
    grpo_scores = [grpo["by_difficulty"].get(k, 0) * 100
                   for k in ("easy", "medium", "hard", "extra")]
    series.append(("GRPO (final)", grpo_scores, "#D35400"))

n = len(series)
width = 0.8 / n   # fit all bars in unit-wide category slot with small margin

fig, ax = plt.subplots(figsize=(12, 6))
offsets = [(i - (n - 1) / 2.0) * width for i in range(n)]
bars_by_series = []
for (label, scores_arr, color), off in zip(series, offsets):
    bars_by_series.append(
        ax.bar(x + off, scores_arr, width, label=label, color=color)
    )
for bar_group in bars_by_series:
    for bar in bar_group:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5,
                    f"{height:.1f}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Execution Accuracy (%)", fontsize=13)
ax.set_title("Execution Accuracy by Query Difficulty", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(difficulties, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 95)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/difficulty_breakdown.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/difficulty_breakdown.png")

# ============================================================
# Chart 3: Exact Match vs Execution Accuracy
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

metrics = ["Update 1\n(Exact Match\n48 samples)", "Update 2\n(Execution Accuracy\n1,034 samples)"]
values = [29.2, 67.8]
bar_colors = ["#E74C3C", "#2ECC71"]

bars = ax.bar(metrics, values, color=bar_colors, width=0.5, edgecolor="white", linewidth=2)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=16)

ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("Same Model, Different Metric", fontsize=15, fontweight="bold")
ax.set_ylim(0, 85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/em_vs_ex.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/em_vs_ex.png")

# ============================================================
# Chart 4: Error Analysis
# ============================================================
preds = zs["predictions"]
failures = [p for p in preds if not p.get("correct", False)]
total_failures = len(failures)

categories = {"Syntax Error": 0, "Execution Error": 0, "Wrong Result": 0}
for f in failures:
    pred_sql = f.get("pred_sql", "")
    pred_err = f.get("pred_error", None)
    if not pred_sql or "SELECT" not in pred_sql.upper():
        categories["Syntax Error"] += 1
    elif pred_err:
        categories["Execution Error"] += 1
    else:
        categories["Wrong Result"] += 1

labels = list(categories.keys())
sizes = list(categories.values())
colors_pie = ["#E74C3C", "#E8A838", "#3498DB"]

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct="%1.1f%%",
                                   colors=colors_pie, textprops={"fontsize": 12},
                                   startangle=90)
for t in autotexts:
    t.set_fontweight("bold")
ax.set_title(f"Zero-shot Failure Analysis (n={total_failures} failures)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/error_analysis.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/error_analysis.png")

# ============================================================
# Chart 5: Data Pipeline Filtering
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))

stages = ["Spider Train\n+ Others", "After Filtering"]
counts = [8659, 7040]

bars = ax.barh(stages, counts, color=["#4A90D9", "#2ECC71"], height=0.5, edgecolor="white")
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2.0,
            f"{count:,}", ha="left", va="center", fontweight="bold", fontsize=14)

ax.annotate(f"-1,619 removed\n(1,616 empty results\n3 errors)",
            xy=(7040, 0), xytext=(5000, 0.7),
            fontsize=11, color="#E74C3C",
            arrowprops=dict(arrowstyle="->", color="#E74C3C"))

ax.set_xlabel("Number of Training Examples", fontsize=12)
ax.set_title("Training Data Preprocessing", fontsize=14, fontweight="bold")
ax.set_xlim(0, 10000)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("figures/data_filtering.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/data_filtering.png")

# ============================================================
# Chart 6: Reward Function Architecture
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Phase 1
components_p1 = ["Execution\nAccuracy", "Syntax\nValid"]
weights_p1 = [1.0, 0.2]
colors_p1 = ["#2ECC71", "#3498DB"]
b1 = ax1.bar(components_p1, weights_p1, color=colors_p1, width=0.5, edgecolor="white")
for bar, w in zip(b1, weights_p1):
    ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02,
             f"{w}", ha="center", va="bottom", fontweight="bold", fontsize=14)
ax1.set_title("Phase 1: Simple Reward", fontsize=12, fontweight="bold")
ax1.set_ylabel("Weight", fontsize=11)
ax1.set_ylim(0, 1.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Phase 2
components_p2 = ["Execution\nAccuracy", "Exec\nSuccess", "Syntax\nValid", "Schema\nF1", "Format"]
weights_p2 = [1.0, 0.2, 0.2, 0.2, 0.1]
colors_p2 = ["#2ECC71", "#27AE60", "#3498DB", "#9B59B6", "#E8A838"]
b2 = ax2.bar(components_p2, weights_p2, color=colors_p2, width=0.6, edgecolor="white")
for bar, w in zip(b2, weights_p2):
    ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02,
             f"{w}", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax2.set_title("Phase 2: Composite Reward", fontsize=12, fontweight="bold")
ax2.set_ylabel("Weight", fontsize=11)
ax2.set_ylim(0, 1.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("figures/reward_architecture.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/reward_architecture.png")

# ============================================================
# Chart 7: SFT Improvement Over Zero-shot
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

categories_imp = ["Easy", "Medium", "Hard", "Extra Hard", "Overall"]
# Values pulled from the same source as Chart 2 — official Spider hardness.
_zs_pd = zs["by_difficulty"]
_sft_pd = sft["by_difficulty"]
_grpo_pd = grpo["by_difficulty"] if grpo is not None else None
# Show GRPO delta over zero-shot if available, else SFT delta
_ref_pd = _grpo_pd if _grpo_pd is not None else _sft_pd
_ref_overall = grpo["overall_ex"] * 100 if grpo is not None else sft["overall_ex"] * 100
_ref_label = "GRPO (final)" if grpo is not None else "SFT (3 epochs)"
zs_vals = [_zs_pd["easy"] * 100, _zs_pd["medium"] * 100, _zs_pd["hard"] * 100, _zs_pd["extra"] * 100, zs["overall_ex"] * 100]
sft_vals = [_ref_pd["easy"] * 100, _ref_pd["medium"] * 100, _ref_pd["hard"] * 100, _ref_pd["extra"] * 100, _ref_overall]
improvements = [sft_vals[i] - zs_vals[i] for i in range(len(zs_vals))]

x = np.arange(len(categories_imp))
bar_colors_imp = ["#27AE60" if v > 0 else ("#95A5A6" if v == 0 else "#E74C3C") for v in improvements]
bars = ax.bar(x, improvements, color=bar_colors_imp, width=0.5, edgecolor="white")

for bar, val in zip(bars, improvements):
    y_pos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 1.5
    ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos,
            f"{val:+.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(categories_imp, fontsize=12)
ax.set_ylabel("EX Change (%)", fontsize=13)
ax.set_title(f"{_ref_label} Improvement Over Zero-shot", fontsize=15, fontweight="bold")
ax.axhline(y=0, color="black", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/sft_improvement.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/sft_improvement.png")

# ============================================================
# Chart 8: SFT delta by difficulty — where SFT wins and loses
# ============================================================
# Uses the official Spider hardness partition. The story under the canonical
# partition: SFT gains on medium and extra-hard, regresses on hard.
diffs_8 = ["Easy", "Medium", "Hard", "Extra Hard"]
zs_vals_8 = [zs["by_difficulty"].get(k, 0) * 100 for k in ["easy", "medium", "hard", "extra"]]
sft_vals_8 = [sft["by_difficulty"].get(k, 0) * 100 for k in ["easy", "medium", "hard", "extra"]]
deltas_8 = [sft_vals_8[i] - zs_vals_8[i] for i in range(4)]

fig, ax = plt.subplots(figsize=(10, 5))
bar_colors_8 = ["#27AE60" if d > 0 else ("#95A5A6" if d == 0 else "#E74C3C") for d in deltas_8]
bars = ax.bar(diffs_8, deltas_8, color=bar_colors_8, width=0.5, edgecolor="white")
for bar, d, zs_v, sft_v in zip(bars, deltas_8, zs_vals_8, sft_vals_8):
    y_pos = bar.get_height() + 0.3 if d >= 0 else bar.get_height() - 1.5
    ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos,
            f"{d:+.1f}%\n({zs_v:.1f}→{sft_v:.1f})",
            ha="center", va="bottom" if d >= 0 else "top", fontsize=10, fontweight="bold")

ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_ylabel("EX Change: SFT (ep 3) − Zero-shot (pp)", fontsize=12)
ax.set_title("Where SFT Wins and Loses (official Spider hardness)",
             fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/sft_delta_by_difficulty.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/sft_delta_by_difficulty.png")

print("\nAll 8 charts generated in figures/")
