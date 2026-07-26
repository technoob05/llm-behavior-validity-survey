# Third-party data and software

The repository's MIT license applies to original code and documentation in this
project. It does not replace the licenses of the sources below.

| Source | Pinned revision | License | Use in this project |
|---|---|---|---|
| [MoralChoice](https://github.com/ninodimontalcino/moralchoice) | `9f1dbced7ecf70e334af9a88c3d93be5af0f37b8` | MIT, as distributed in `LICENSE.md` | Released model-response files used for the dilemma and clear-answer re-analyses |
| [PromptEval MMLU correctness](https://huggingface.co/datasets/PromptEval/PromptEval_MMLU_correctness) | `1639d5ea14c362f6964f260ae81bd903af760187` | MIT, as declared in the dataset card | Released correctness arrays used for the capability re-analysis |
| [MT-Bench release](https://huggingface.co/spaces/lmsys/mt-bench/blob/main/data/mt_bench/model_judgment/gpt-4_pair.jsonl) | `6e465b26cb18b64e48b3858d54ac655736cf07b6` | Apache-2.0 via the FastChat project | Released pairwise-judgment records used for the open-ended example; SHA-256 is verified on download |

The ARR software archive contains analysis code and small derived outputs, not
copies of the large third-party response corpora. The retrieval script fetches
those corpora from their original public releases. Any redistribution in a later
data archive must preserve the source license and attribution.
