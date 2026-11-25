# Agent 4 - one command per purpose. Heavy steps are submitted to Slurm by the controller.
PY := /vast/som/yunligrp/users/ygao/conda/envs/agent3/bin/python
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)
export PYTHONNOUSERSITE := 1
export PYTHONDONTWRITEBYTECODE := 1
export AGENT4_ROOT := $(CURDIR)

.PHONY: all replay status test reports package

all:            ## run both tasks end to end; finished steps are skipped
	$(PY) -B scripts/controller.py all

replay:         ## same, but LLM answers must come from llm_cache/ (no keys, no login, no network)
	A4_REPLAY_ONLY=1 $(PY) -B scripts/controller.py all

status:         ## which stages exist for each task
	$(PY) -B scripts/controller.py status

test:           ## fast checks: isolation of hidden data, decision rules, model interfaces
	$(PY) -B -m pytest -q tests -p no:cacheprovider

reports:        ## project report and slides from saved results
	$(PY) -B scripts/build_project_report.py

package:        ## submission archive without raw data, encoder weights or encoder features
	$(PY) -B scripts/package_submission.py
