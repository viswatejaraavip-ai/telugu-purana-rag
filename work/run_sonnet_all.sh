#!/bin/bash
# Sonnet 5 on every chapter lacking a card, main cards/ folder, total cap $17.
cd ~/telugu-purana-corpus
set -a; source ~/telugu-rag/.env; set +a
CAP=17
.venv/bin/python scripts/gen_cards.py bhagavata --model claude-sonnet-5 --workers 6 --max-usd $CAP 2>&1 | grep -v NotOpenSSL | tee work_sonnet_bhagavata.log
SPENT=$(grep -oE 'total=\$[0-9.]+' work_sonnet_bhagavata.log | tail -1 | tr -d 'total=$')
LEFT=$(python3 -c "print(max(0, round($CAP - ${SPENT:-0}, 2)))")
echo "bhagavata spent \$$SPENT, cap left \$$LEFT"
if [ "$(python3 -c "print(1 if $LEFT > 0.5 else 0)")" = "1" ]; then
  .venv/bin/python scripts/gen_cards.py vishnu --model claude-sonnet-5 --workers 6 --max-usd $LEFT 2>&1 | grep -v NotOpenSSL | tee work_sonnet_vishnu.log
fi
echo "ALL DONE: bhagavata=$(ls cards/bhagavata | wc -l)/335 vishnu=$(ls cards/vishnu 2>/dev/null | wc -l)/126"
