"""CLI: python app/ask.py "ప్రశ్న" [--json]   (run with ~/telugu-rag/.venv)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import answer
args = [a for a in sys.argv[1:] if a != "--json"]
out = answer(" ".join(args))
if "--json" in sys.argv:
    keep = {k: out.get(k) for k in ("question", "question_norm", "route", "works", "query", "answer", "citations", "verification", "grade_reason")}
    keep["chapters"] = [dict(adhyaya_id=c["adhyaya_id"], citation_te=c["citation_te"], title_te=c["card"]["title_te"], score=c["_score"]) for c in out.get("cards", [])]
    print(json.dumps(keep, ensure_ascii=False, indent=1))
else:
    print(out["answer"]); print()
    for c in out.get("citations", []): print(f"• {c['citation_te']}: {c['tatparyam_te']}")
