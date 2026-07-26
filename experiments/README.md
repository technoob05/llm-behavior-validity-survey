# Prepared experiments

`repeated_judge_mtbench.py` closes the main remaining empirical gap. It creates independent judge repeats for both presentation orders of each MT-Bench pair. Those repeats are required before estimating broad $\phi$ for open-ended judging; the released MT-Bench labels support only strict $\phi$ and observed disagreement.

`order_sensitivity_case_study.py` is the self-contained T2 case study for
option-order sensitivity. All runnable experiments now have one home.

## Local bookkeeping smoke test

```powershell
python experiments\repeated_judge_mtbench.py --input analysis\mtbench\gpt4_pair.jsonl --max-items 8 --repeats 2 --backend mock --output experiments\outputs\smoke.jsonl
```

The mock backend is only for testing the data schema. It is not scientific evidence.

## GPU experiment

Use `--backend transformers` with an Apache-2.0 judge such as `Qwen/Qwen2.5-7B-Instruct`, at least three independent repeats, and raw output retention. Start with a 200-item L4 pilot, then run all 4,796 paired comparisons. HF Jobs cannot read a local file, so the exact frozen input must be available through `--input-url` in the anonymous artifact.
