@echo off
REM Nyaya Moot Court - double-click to start (Windows)
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python -m pip install -q -r requirements-legalbert.txt || echo LegalBERT libraries not installed - app will run without argument tags.
python server.py %*
pause
