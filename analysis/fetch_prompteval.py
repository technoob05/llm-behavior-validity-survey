"""Re-download the PromptEval MMLU correctness matrices we re-analyse.

Each parquet is items x 100 meaning-preserving prompt templates, entries 0/1,
for one (subject, model) pair. We keep a spread of subjects and models rather
than the whole 857-file release, which is enough for the variance work and keeps
the download small.
"""
import io
import os
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "prompteval"
OUT.mkdir(exist_ok=True)
REVISION = "1639d5ea14c362f6964f260ae81bd903af760187"
BASE = (
    "https://huggingface.co/datasets/PromptEval/"
    f"PromptEval_MMLU_correctness/resolve/{REVISION}/"
)

SUBJECTS = [
    "abstract_algebra", "college_mathematics", "high_school_psychology",
    "moral_scenarios", "professional_law", "world_religions",
    "us_foreign_policy", "clinical_knowledge", "formal_logic",
    "high_school_world_history", "philosophy", "econometrics",
]
MODELS = [
    "meta_llama_llama_3_8b_instruct", "meta_llama_llama_3_70b_instruct",
    "mistralai_mistral_7b_instruct_v0_2", "google_flan_t5_xl",
    "google_gemma_7b_it", "codellama_codellama_34b_instruct",
    "tiiuae_falcon_40b", "mistralai_mixtral_8x7b_instruct_v0_1",
]


def grab(subject, model):
    dest = OUT / ("%s__%s.parquet" % (subject, model))
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    url = BASE + "%s/%s-00000-of-00001.parquet" % (subject, model)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except Exception as e:
        return "MISS (%s)" % type(e).__name__
    dest.write_bytes(data)
    return "%.0f KB" % (len(data) / 1024)


if __name__ == "__main__":
    ok = 0
    for s in SUBJECTS:
        for m in MODELS:
            st = grab(s, m)
            if "MISS" not in st:
                ok += 1
            print("%-28s %-38s %s" % (s, m, st))
    print("\nfiles on disk:", len(list(OUT.glob("*.parquet"))), "| ok:", ok)
