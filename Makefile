.PHONY: test sync
test:
	python -m pytest -q
sync:
	python scripts/sync_agent_files.py
