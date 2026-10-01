import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.vector_store import VectorStoreService
from utils.path_tool import get_abs_path


DATASET_PATH = Path(get_abs_path("evaluation/questions.json"))
RESULTS_DIR = Path(get_abs_path("evaluation/results"))
K_VALUES = (1,3,5)


def load_questions() -> list[dict[str, object]]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    questions = dataset.get("questions", [])
    if not questions:
        raise ValueError(f"评估问题集为空：{DATASET_PATH}")

    for question in questions:
        if not question.get("question") or not question.get("relevant_sources"):
            raise ValueError(f"问题缺少 question 或 relevant_sources：{question}")

    return questions


def evaluate(vector_store , questions):
    reports = {}

    for k in K_VALUES:
        retriever = vector_store.get_retriever(k=k)
        question_results = []
        recall_scores = []
        hit_scores = []
        latencies_ms = []
        empty_results = 0

        for item in questions:
            question = str(item["question"])
            relevant_sources = {Path(str(source)).name for source in item["relevant_sources"]}

            started_at = perf_counter()
            documents = retriever.invoke(question)
            latency_ms = (perf_counter() - started_at) * 1000

            retrieved_sources = {
                Path(str(document.metadata.get("source", ""))).name
                for document in documents
                if document.metadata.get("source")
            }
            matched_sources = sorted(relevant_sources & retrieved_sources)
            recall_at_k = len(matched_sources) / len(relevant_sources)
            hit_at_k = int(bool(matched_sources))

            recall_scores.append(recall_at_k)
            hit_scores.append(hit_at_k)
            latencies_ms.append(latency_ms)
            empty_results += int(not documents)
            question_results.append({
                "id": item.get("id"),
                "question": question,
                "relevant_sources": sorted(relevant_sources),
                "retrieved_sources": sorted(retrieved_sources),
                "matched_sources": matched_sources,
                "recall_at_k": recall_at_k,
                "hit_at_k": hit_at_k,
                "latency_ms": round(latency_ms, 2),
            })

        question_count = len(questions)
        reports[str(k)] = {
            "k": k,
            "question_count": question_count,
            "source_recall_at_k": sum(recall_scores) / question_count,
            "hit_rate_at_k": sum(hit_scores) / question_count,
            "average_latency_ms": sum(latencies_ms) / question_count,
            "empty_result_rate": empty_results / question_count,
            "questions": question_results,
        }

    return reports


def main() -> None:
    questions = load_questions()
    vector_store = VectorStoreService()
    metrics = evaluate(vector_store, questions)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "question_count": len(questions),
        "metrics_by_k": metrics,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "retrieval_results.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"评估问题数：{len(questions)}")
    for result in metrics.values():
        print(
            f"k={result['k']}: "
            f"Source Recall@k={result['source_recall_at_k']:.1%}, "
            f"Hit@k={result['hit_rate_at_k']:.1%}, "
            f"平均耗时={result['average_latency_ms']:.1f} ms, "
            f"空结果率={result['empty_result_rate']:.1%}"
        )
    print(f"逐题结果已保存到：{result_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
