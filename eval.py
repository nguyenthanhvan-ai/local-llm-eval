"""
Compare local LLMs served by Ollama on a small grounded-QA test set.

Checks, per model:
  - answer accuracy on questions that the context can answer
  - refusal rate on questions the context cannot answer (hallucination check)
  - latency per answer

Usage:
  python eval.py                 # run every model in MODELS
  python eval.py --models qwen3:8b
"""
import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
TESTSET = Path(__file__).parent / "testset.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

MODELS = {
    "qwen3-8b": "qwen3:8b",
    "stheno-v3.2-8b": "hf.co/bartowski/L3-8B-Stheno-v3.2-GGUF:Q4_K_M",
    "ohashi-nemo-12b": "hf.co/featherless-ai-quants/Delta-Vector-Ohashi-NeMo-12B-GGUF:Q4_K_M",
}

SYSTEM_PROMPT = (
    "Chỉ trả lời dựa trên tài liệu được cung cấp. "
    "Nếu tài liệu không chứa thông tin cần thiết, hãy trả lời đúng một câu: "
    "\"Tài liệu không có thông tin này.\""
)

REFUSAL_PATTERNS = [
    r"không có thông tin",
    r"không đề cập",
    r"không được đề cập",
    r"không có trong tài liệu",
    r"không nêu",
    r"không tìm thấy",
    r"không biết",
]


def load_testset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ask(model: str, item: dict) -> tuple[str, float]:
    prompt = f"Tài liệu:\n{item['context']}\n\nCâu hỏi: {item['question']}"
    t0 = time.perf_counter()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,  # turns off Qwen3 reasoning output; ignored by models without it
            "options": {"temperature": 0, "num_predict": 300},
        },
        timeout=300,
    )
    resp.raise_for_status()
    answer = resp.json()["message"]["content"].strip()
    return answer, time.perf_counter() - t0


def is_refusal(answer: str) -> bool:
    return any(re.search(p, answer, re.IGNORECASE) for p in REFUSAL_PATTERNS)


def score(item: dict, answer: str) -> bool:
    if item["type"] == "unanswerable":
        return is_refusal(answer)
    # answerable: every expected regex must match, and the model must not refuse
    if is_refusal(answer):
        return False
    return all(re.search(p, answer, re.IGNORECASE) for p in item["expect"])


def run_model(label: str, model: str, items: list[dict], writer) -> dict:
    rows = []
    for item in items:
        try:
            answer, latency = ask(model, item)
            passed = score(item, answer)
            error = ""
        except requests.RequestException as exc:
            answer, latency, passed, error = "", 0.0, False, str(exc)
        row = {
            "model": label,
            "id": item["id"],
            "type": item["type"],
            "passed": passed,
            "latency_s": round(latency, 2),
            "answer": answer.replace("\n", " "),
            "error": error,
        }
        writer.writerow(row)
        rows.append(row)
        print(f"[{label}] {item['id']:<4} {'PASS' if passed else 'FAIL'}  {latency:5.1f}s")
    return summarize(label, rows)


def summarize(label: str, rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] == "answerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    latencies = [r["latency_s"] for r in rows if not r["error"]]
    return {
        "model": label,
        "accuracy_answerable": sum(r["passed"] for r in answerable) / max(len(answerable), 1),
        "refusal_rate_unanswerable": sum(r["passed"] for r in unanswerable) / max(len(unanswerable), 1),
        "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
        "errors": sum(1 for r in rows if r["error"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(MODELS.values()),
                        help="Ollama model names to test (default: all in MODELS)")
    args = parser.parse_args()

    items = load_testset(TESTSET)
    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / "answers.csv"
    summaries = []

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["model", "id", "type", "passed", "latency_s", "answer", "error"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model in args.models:
            label = next((k for k, v in MODELS.items() if v == model), model)
            summaries.append(run_model(label, model, items, writer))

    with (RESULTS_DIR / "summary.md").open("w", encoding="utf-8") as f:
        f.write("| Model | Accuracy (answerable) | Refusal rate (unanswerable) | Median latency (s) | Errors |\n")
        f.write("|---|---|---|---|---|\n")
        for s in summaries:
            f.write(f"| {s['model']} | {s['accuracy_answerable']:.0%} | "
                    f"{s['refusal_rate_unanswerable']:.0%} | {s['median_latency_s']} | {s['errors']} |\n")
    print(f"\nWrote {out_csv} and results/summary.md")


if __name__ == "__main__":
    main()
