"""
Case study for the survey "Can We Believe What Large Language Models Do?"
Reproduces the T2 order-and-presentation-bias threat (Zheng et al. 2024,
selection bias) with a single frozen open model, self-contained.

WHAT IT MEASURES
  For each multiple-choice item we run all permutations of the option order,
  read the answer-token logits, and record which option the model picks.
  From that we compute three numbers that the survey cites as the T2 signal:
    (1) flip rate      = fraction of items whose predicted answer changes
                         across orderings (behaviour is order-dependent).
    (2) position bias  = how often each slot (1st, 2nd, ...) is chosen,
                         marginalised over content; a flat model would be
                         uniform, a biased one spikes on one slot.
    (3) accuracy gap   = best-ordering accuracy minus worst-ordering accuracy
                         (the swing a single-order study would have reported).

WHY THIS DESIGN
  The items below are neutral, unambiguous general-knowledge questions with a
  single correct answer, embedded so the script is fully self-contained and
  carries no dataset licence. The point is methodological (does the answer
  depend on presentation), not a benchmark score, so a small hand-built set
  is sufficient and avoids contamination confounds.

HOW TO RUN (offline, Kaggle or local GPU)
  Add Qwen2.5-7B-Instruct as a Kaggle model mount (or set MODEL_DIR), then:
      python experiments/order_sensitivity_case_study.py
  Outputs: results_order_sensitivity.json  and  fig_order_sensitivity.pdf/png

NOTE
  Logit-only, no generation. One model resident. ASCII-only, self-contained.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import glob
import json
import math
import itertools
import argparse

import numpy as np


# --------------------------------------------------------------------------
# Self-contained item set: (question, options_in_canonical_order, gold_index)
# Neutral, single-answer, general knowledge. Gold is the index in the
# canonical option list; the script permutes the presented order internally.
# --------------------------------------------------------------------------
ITEMS = [
    ("What is the capital of Japan?",
     ["Tokyo", "Seoul", "Beijing", "Bangkok"], 0),
    ("Which planet is closest to the Sun?",
     ["Mercury", "Venus", "Mars", "Earth"], 0),
    ("How many sides does a hexagon have?",
     ["Six", "Five", "Seven", "Eight"], 0),
    ("What gas do plants primarily absorb for photosynthesis?",
     ["Carbon dioxide", "Oxygen", "Nitrogen", "Hydrogen"], 0),
    ("Who wrote the play 'Romeo and Juliet'?",
     ["William Shakespeare", "Charles Dickens", "Mark Twain", "Jane Austen"], 0),
    ("What is the chemical symbol for gold?",
     ["Au", "Ag", "Gd", "Go"], 0),
    ("Which ocean is the largest by area?",
     ["Pacific", "Atlantic", "Indian", "Arctic"], 0),
    ("What is the freezing point of water in degrees Celsius?",
     ["Zero", "Thirty-two", "One hundred", "Minus ten"], 0),
    ("Which language has the most native speakers worldwide?",
     ["Mandarin Chinese", "English", "Spanish", "Hindi"], 0),
    ("What is the square root of 144?",
     ["Twelve", "Ten", "Fourteen", "Sixteen"], 0),
    ("Which organ pumps blood through the human body?",
     ["Heart", "Liver", "Lung", "Kidney"], 0),
    ("What is the largest mammal on Earth?",
     ["Blue whale", "African elephant", "Giraffe", "Hippopotamus"], 0),
    ("In which continent is the Sahara Desert located?",
     ["Africa", "Asia", "Australia", "South America"], 0),
    ("What is the primary language spoken in Brazil?",
     ["Portuguese", "Spanish", "French", "Italian"], 0),
    ("How many minutes are there in one hour?",
     ["Sixty", "Thirty", "Ninety", "One hundred"], 0),
]

LETTERS = ["A", "B", "C", "D", "E", "F"]


def find_model_dir(user_path):
    if user_path and os.path.isdir(user_path):
        return user_path
    patterns = [
        "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/*",
        "/kaggle/input/**/qwen2.5*7b*instruct*/**",
        "/kaggle/input/**/*7b-instruct*/**",
        os.path.expanduser("~/models/Qwen2.5-7B-Instruct"),
    ]
    for pat in patterns:
        hits = [p for p in glob.glob(pat, recursive=True)
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json"))]
        if hits:
            return sorted(hits)[0]
    raise SystemExit("Model dir not found. Pass --model_dir /path/to/Qwen2.5-7B-Instruct")


def build_prompt(question, presented_options):
    lines = [question, ""]
    for i, opt in enumerate(presented_options):
        lines.append("%s. %s" % (LETTERS[i], opt))
    lines.append("")
    lines.append("Answer with a single letter.")
    return "\n".join(lines)


def load_model(model_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda",
        low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="sdpa")
    model.eval()
    return tok, model


def letter_token_ids(tok, n_options):
    # id of the first sub-token of each answer letter, robust to leading space
    ids = []
    for i in range(n_options):
        cand = []
        for form in (LETTERS[i], " " + LETTERS[i]):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                cand.append(enc[0])
        ids.append(cand[0] if cand else None)
    return ids


def answer_logits(tok, model, prompt, n_options):
    import torch
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc)
    last = out.logits[0, -1, :].float().cpu().numpy()
    ids = letter_token_ids(tok, n_options)
    scores = np.array([last[i] if i is not None else -1e9 for i in ids])
    return scores


def run(model_dir, out_json, out_fig):
    tok, model = load_model(model_dir)

    n_slot_counts = None
    per_item = []
    order_correct = {}   # ordering-signature -> list of correctness per item

    for q, opts, gold in ITEMS:
        n = len(opts)
        if n_slot_counts is None:
            n_slot_counts = np.zeros(n)
        perms = list(itertools.permutations(range(n)))
        picks_canonical = []      # predicted answer as a canonical option index
        for perm in perms:
            presented = [opts[j] for j in perm]
            scores = answer_logits(tok, model, build_prompt(q, presented), n)
            slot = int(np.argmax(scores))            # chosen slot (0..n-1)
            n_slot_counts[slot] += 1
            canonical_pick = perm[slot]              # map slot back to option
            picks_canonical.append(canonical_pick)
            sig = "".join(str(p) for p in perm)
            order_correct.setdefault(sig, []).append(int(canonical_pick == gold))

        picks_canonical = np.array(picks_canonical)
        flipped = int(len(np.unique(picks_canonical)) > 1)
        acc = float(np.mean(picks_canonical == gold))
        per_item.append({"question": q, "flipped": flipped, "acc_over_orders": acc})

    flip_rate = float(np.mean([it["flipped"] for it in per_item]))
    position_bias = (n_slot_counts / n_slot_counts.sum()).tolist()
    order_acc = {sig: float(np.mean(v)) for sig, v in order_correct.items()}
    acc_gap = float(max(order_acc.values()) - min(order_acc.values()))

    result = {
        "model_dir": model_dir,
        "n_items": len(ITEMS),
        "n_orderings_per_item": int(math.factorial(len(ITEMS[0][1]))),
        "flip_rate": flip_rate,
        "position_bias": position_bias,
        "accuracy_gap_best_worst": acc_gap,
        "per_item": per_item,
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print("flip_rate=%.3f  acc_gap=%.3f  position_bias=%s"
          % (flip_rate, acc_gap, ["%.2f" % b for b in position_bias]))

    make_figure(result, out_fig)
    return result


def make_figure(result, out_fig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INDIGO = "#4338CA"
    ROSE = "#E11D48"
    INK = "#1E293B"
    GRID = "#E2E8F0"

    pb = result["position_bias"]
    n = len(pb)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))

    ax0 = ax[0]
    xs = list(range(n))
    ax0.bar(xs, pb, color=INDIGO, width=0.62, zorder=3)
    ax0.axhline(1.0 / n, color=ROSE, lw=1.6, ls="--", zorder=4,
                label="unbiased (%.2f)" % (1.0 / n))
    ax0.set_xticks(xs)
    ax0.set_xticklabels(["slot %d" % (i + 1) for i in range(n)])
    ax0.set_ylabel("share of picks")
    ax0.set_title("Position bias: which slot is chosen", color=INK, fontsize=11)
    for s in ("top", "right"):
        ax0.spines[s].set_visible(False)
    ax0.grid(axis="y", color=GRID, alpha=0.6, zorder=0)
    ax0.legend(frameon=False, fontsize=8)

    ax1 = ax[1]
    vals = [result["flip_rate"], result["accuracy_gap_best_worst"]]
    labels = ["answer\nflip rate", "accuracy gap\n(best - worst order)"]
    bars = ax1.bar([0, 1], vals, color=[ROSE, INDIGO], width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, "%.2f" % v,
                 ha="center", fontsize=11, fontweight="bold", color=INK)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Order sensitivity of one frozen model", color=INK, fontsize=11)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.grid(axis="y", color=GRID, alpha=0.6, zorder=0)

    fig.suptitle("Reproducing T2 (order and presentation bias): "
                 "the same items, only the option order changes",
                 fontsize=11.5, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_fig + "." + ext, dpi=200, bbox_inches="tight")
    print("figure written to %s.pdf/.png" % out_fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--out_json", default="results_order_sensitivity.json")
    ap.add_argument("--out_fig", default="fig_order_sensitivity")
    a = ap.parse_args()
    md = find_model_dir(a.model_dir)
    print("using model:", md)
    run(md, a.out_json, a.out_fig)
