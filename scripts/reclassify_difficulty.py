"""Re-label predictions with Spider's official hardness classifier.

Our src/eval/run_eval.py uses a heuristic classify_difficulty() that
produces a different partition than Spider's canonical evaluator
(e.g. extra-hard bucket collapses to n=6 instead of 250). This script
vendors Spider's evaluation.py (in third_party/spider/) and re-buckets
our existing prediction JSONs using eval_hardness().

No model inference, no GPU — pure AST re-labeling from dev.json.
"""
import json
import os
import sys
from collections import defaultdict

SPIDER_EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "third_party", "spider")
sys.path.insert(0, SPIDER_EVAL_DIR)

from evaluation import Evaluator  # noqa: E402

DEV_PATH = "data/spider_data/spider_data/dev.json"
EVAL_DIR = "results"
OUT_SUFFIX = "_official_hardness"


def load_official_hardness(dev_path: str) -> list[str]:
    """Apply Spider's eval_hardness to every dev example (by index)."""
    dev = json.load(open(dev_path))
    ev = Evaluator()
    return [ev.eval_hardness(ex["sql"]) for ex in dev]


def recompute(results_path: str, hardness: list[str]) -> dict:
    data = json.load(open(results_path))
    preds = data["predictions"]
    assert len(preds) == len(hardness), f"{len(preds)} != {len(hardness)}"

    totals = defaultdict(int)
    correct = defaultdict(int)
    for p, h in zip(preds, hardness):
        p["difficulty_official"] = h
        totals[h] += 1
        if p.get("correct"):
            correct[h] += 1

    by_diff = {k: correct[k] / totals[k] for k in totals}
    data["by_difficulty_official"] = by_diff
    data["counts_official"] = dict(totals)
    data["correct_by_difficulty_official"] = dict(correct)
    return data


def main():
    hardness = load_official_hardness(DEV_PATH)
    from collections import Counter
    print("Official Spider dev partition:")
    c = Counter(hardness)
    for k in ["easy", "medium", "hard", "extra"]:
        print(f"  {k}: {c[k]}")
    print(f"  total: {sum(c.values())}")
    print()

    # Discover every eval_dev_*.json that isn't already an official-hardness
    # output so new result files (e.g. GRPO checkpoints) get picked up
    # automatically without editing the script.
    import glob
    candidates = sorted(glob.glob(os.path.join(EVAL_DIR, "eval_dev_*.json")))
    targets = [p for p in candidates if OUT_SUFFIX not in os.path.basename(p)]
    for path in targets:
        name = os.path.basename(path)[len("eval_dev_"):-len(".json")]
        d = recompute(path, hardness)
        print(f"{name}:")
        print(f"  overall: {d['correct']}/{d['total']} = {d['correct']/d['total']*100:.2f}%")
        for k in ["easy", "medium", "hard", "extra"]:
            n = d["counts_official"][k]
            c_ = d["correct_by_difficulty_official"][k]
            print(f"  {k:6s}: {c_:3d}/{n:3d} = {c_/n*100:.2f}%")
        out = path.replace(".json", f"{OUT_SUFFIX}.json")
        json.dump(d, open(out, "w"), indent=2)
        print(f"  wrote {out}")
        print()


if __name__ == "__main__":
    main()
