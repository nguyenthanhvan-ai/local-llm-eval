# Local LLM Grounded-QA Evaluation

Compares three locally-hosted language models served by [Ollama](https://ollama.com) on a small
grounded question-answering test set. The goal is to measure two things that matter for real
document-QA apps: whether a model answers correctly from the provided context, and whether it
refuses when the context does not contain the answer (a hallucination check).

## Models tested

| Label | Ollama model name |
|---|---|
| qwen3-8b | `qwen3:8b` |
| stheno-v3.2-8b | `hf.co/bartowski/L3-8B-Stheno-v3.2-GGUF:Q4_K_M` |
| ohashi-nemo-12b | `hf.co/featherless-ai-quants/Delta-Vector-Ohashi-NeMo-12B-GGUF:Q4_K_M` |

## Test set

`testset.jsonl` has 16 items in Vietnamese:
- **11 answerable** items, built from two short fictional/neutral passages (a company profile and a Python reference note). Each has regex patterns that the answer must match.
- **5 unanswerable** items, where the context does not contain the answer. The correct behaviour is to say the document has no such information.

The company profile is fictional so that models cannot rely on memorised facts.

## How to run

```bash
# 1. Install Ollama and pull the models you want to test
ollama pull qwen3:8b
ollama pull hf.co/bartowski/L3-8B-Stheno-v3.2-GGUF:Q4_K_M

# 2. Install the only dependency
pip install -r requirements.txt

# 3. Run the evaluation
python eval.py
```

Outputs:
- `results/answers.csv`: every answer, whether it passed, and its latency.
- `results/summary.md`: accuracy, refusal rate and median latency per model.

Run a subset with `python eval.py --models qwen3:8b`.

## Scoring

- **Answerable:** the answer must match every expected pattern and must not be a refusal.
- **Unanswerable:** the answer must be a refusal (matches one of the refusal patterns in `eval.py`).

Scoring is keyword- and pattern-based, so it is a rough check, not a full evaluation of answer quality.

## Results

See [`results/summary.md`](results/summary.md) after running the script.

## Limitations

- 16 questions is a small sample. Treat the numbers as a first look, not a benchmark.
- Pattern matching can mark a correct answer as wrong if it uses unexpected wording.
- All models run with temperature 0 on the same machine. Latency depends on hardware.

## Possible next steps

- Add more documents and harder unanswerable questions.
- Replace pattern matching with an LLM-as-judge or human review on a sample.
- Add a retrieval step so the models can search a larger document set.
