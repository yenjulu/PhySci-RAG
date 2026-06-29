import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8")

from physci_rag import PhySciRAG


def main() -> None:
  parser = argparse.ArgumentParser(description="Query the PhySciBench RAG system.")
  parser.add_argument("question", nargs="?", help="Free-form question to ask.")
  parser.add_argument("--id", dest="record_id", help="Benchmark record id, e.g. physci-001.")
  parser.add_argument("--top-k", type=int, default=6, help="Number of chunks to retrieve.")
  parser.add_argument(
    "--retrieve-only",
    action="store_true",
    help="Return retrieved passages without LLM generation.",
  )
  parser.add_argument(
    "--list",
    action="store_true",
    help="List sample benchmark questions.",
  )
  parser.add_argument("--type", dest="task_type", help="Filter listed questions by task type.")
  args = parser.parse_args()

  rag = PhySciRAG()

  if args.list:
    for record in rag.list_questions(task_type=args.task_type, limit=15):
      print(f"{record.id} [{record.type}] {record.question[:120]}...")
    return

  if not args.question and not args.record_id:
    parser.error("Provide a question, --id, or --list.")

  result = rag.query(
    question=args.question or "",
    top_k=args.top_k,
    record_id=args.record_id,
    use_llm=not args.retrieve_only,
  )

  print(f"Question: {result.question}\n")
  if result.task_type:
    print(f"Task type: {result.task_type}")
  print("Answer:")
  print(result.answer)
  print("\nSources:")
  for idx, item in enumerate(result.contexts, start=1):
    print(
      f"  {idx}. {item.chunk.source_file} "
      f"(page {item.chunk.page}, score={item.score:.3f})"
    )
  if result.ground_truth:
    print("\nGround truth (benchmark):")
    print(result.ground_truth[:1200])


if __name__ == "__main__":
  main()
