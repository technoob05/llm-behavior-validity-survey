# /// script
# dependencies = ["torch>=2.2", "transformers>=4.45", "accelerate>=1.0", "requests>=2.31"]
# ///
"""Collect independent, order-balanced MT-Bench judge decisions.

HF Jobs example (after exposing a frozen input URL in an anonymous artifact):
hf jobs uv run experiments/repeated_judge_mtbench.py --flavor l4x1 --timeout 8h -- --input-url https://.../gpt4_pair.jsonl --backend transformers --model Qwen/Qwen2.5-7B-Instruct --repeats 3 --output repeated_judgments.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

INSTRUCTION = "You are an impartial evaluator. Compare A and B on the user's request. Return exactly [[A]], [[B]], or [[TIE]].\n\n"

def args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path)
    src.add_argument("--input-url")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("transformers", "mock"), default="transformers")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260726)
    p.add_argument("--max-items", type=int)
    return p.parse_args()

def load(a):
    if a.input_url:
        import requests
        response = requests.get(a.input_url, timeout=120)
        response.raise_for_status()
        lines = response.text.splitlines()
    else:
        lines = a.input.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return rows[:a.max_items] if a.max_items else rows

def make_decider(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    def decide(prompt, seed):
        torch.manual_seed(seed)
        batch = tokenizer(prompt, return_tensors="pt").to(model.device)
        result = model.generate(**batch, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=12)
        text = tokenizer.decode(result[0][batch.input_ids.shape[1]:], skip_special_tokens=True)
        upper = text.upper()
        winner = "A" if "[[A]]" in upper else "B" if "[[B]]" in upper else "TIE" if "[[TIE]]" in upper else "UNPARSED"
        return winner, text
    return decide

def main():
    a = args()
    if a.repeats < 2: raise ValueError("At least two independent repeats are required.")
    rows = load(a)
    decide = make_decider(a.model) if a.backend == "transformers" else None
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            item_id = f"{row['question_id']}|{row['model_1']}|{row['model_2']}|{row['turn']}"
            for order, field in (("g1", "g1_user_prompt"), ("g2", "g2_user_prompt")):
                prompt = INSTRUCTION + row[field]
                for repeat in range(a.repeats):
                    seed = a.seed + 1000003 * i + 101 * repeat + (0 if order == "g1" else 1)
                    if a.backend == "mock":
                        winner = ("A", "B", "TIE")[int(hashlib.sha256(f"{seed}|{prompt}".encode()).hexdigest(), 16) % 3]
                        raw = "MOCK_BACKEND_NOT_EVIDENCE"
                    else:
                        winner, raw = decide(prompt, seed)
                    record = {"item_id": item_id, "question_id": row["question_id"], "model_1": row["model_1"], "model_2": row["model_2"], "turn": row["turn"], "order": order, "repeat": repeat, "seed": seed, "judge_model": a.model, "backend": a.backend, "winner": winner, "raw_generation": raw}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows) * 2 * a.repeats} decisions to {a.output}")

if __name__ == "__main__": main()
