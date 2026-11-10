PY ?= .venv/bin/python
SEED ?= 42

.PHONY: setup data report docs baselines test all

setup:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

data:
	PYTHONPATH=src $(PY) scripts/generate_data.py --seed $(SEED)

report:
	PYTHONPATH=src $(PY) -m movrec.data_quality

docs:
	PYTHONPATH=src $(PY) -c "from movrec.assumptions import load_assumptions, render_markdown; open('docs/ASSUMPTIONS.md','w').write(render_markdown(load_assumptions()))"

baselines:
	PYTHONPATH=src $(PY) -m movrec.run_baselines

test:
	$(PY) -m pytest -q

all: data report docs baselines test
