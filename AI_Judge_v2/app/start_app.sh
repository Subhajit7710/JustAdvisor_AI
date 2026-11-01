#!/usr/bin/env bash
# Nyaya Moot Court - start (Linux / macOS):  bash start_app.sh   (add --demo to try without models)
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt 2>/dev/null || python3 -m pip install -q --user -r requirements.txt
python3 -m pip install -q -r requirements-legalbert.txt 2>/dev/null || python3 -m pip install -q --user -r requirements-legalbert.txt \
  || echo "LegalBERT libraries not installed - app will run without argument tags."
python3 server.py "$@"
