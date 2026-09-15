#!/bin/bash
# usage: ask_space.sh "question"   -> prints the three outputs of /ask
TOK=$(cat ~/.cache/huggingface/token); BASE="https://amrithatejaswiservices-telugu-purana-rag.hf.space/gradio_api/call/ask"
Q=$(python3 -c "import json,sys; print(json.dumps({'data':[sys.argv[1]]}, ensure_ascii=False))" "$1")
EID=$(curl -s -m 30 -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" -d "$Q" "$BASE" | python3 -c "import json,sys; print(json.load(sys.stdin)['event_id'])")
curl -sN -m 400 -H "Authorization: Bearer $TOK" "$BASE/$EID" | grep '^data:' | tail -1 | sed 's/^data: //' | python3 -c "
import json,sys; d=json.load(sys.stdin)
for lbl, x in zip(('ANSWER','CITATIONS','CHAPTERS'), d): print(f'=== {lbl}\n{x}\n')"
