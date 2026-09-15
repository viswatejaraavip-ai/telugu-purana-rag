#!/bin/bash
cd ~/telugu-purana-corpus
set -a; source ~/telugu-rag/.env; set +a
while true; do
  .venv/bin/python scripts/gen_cards_batch.py collect 2>&1 | grep -v NotOpenSSL | tee -a work/poll_batches.log
  if ! grep -qE '"status": "in_progress"|"status": "canceling"' work/api_batches.json; then echo "ALL BATCHES ENDED" | tee -a work/poll_batches.log; break; fi
  sleep 300
done
