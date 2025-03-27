

help:
	@echo 'make install:                                Install package, hook, notebooks and gdslib'
	@echo 'make test:                                   Run tests with pytest'
	@echo 'make test-force:                             Rebuilds regression test'
	@echo 'make docs:                                   Build the docs and place them in ./site'
	@echo 'make docs-serve:                             mkdocs serve the docs'
	@echo 'make release-dr VERSION=MAJOR.MINOR.PATCH:   Dry run for new release with version number v${MAJOR}.${MINOR}.${PATCH}'
	@echo 'make release VERSION=MAJOR.MINOR.PATCH:      Dry run for new release with version number v${MAJOR}.${MINOR}.${PATCH}'

uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

install:
	uv sync --extra docs --extra dev
	uv pip install -e .

dev:
	uv sync --all-extras
	uv pip install -e .
	uv run pre-commit install

test-venv:
	uv sync --all-extras
	uv pip install -e .

docs-clean:
	rm -rf site

docs:
	mkdocs build -f docs/mkdocs.yml

docs-serve:
	mkdocs serve -f docs/mkdocs.yml

test:
	uv run --extra ci --isolated pytest -s -n logical

test-min:
	uv run --isolated --no-cache --no-sync --extra ci --with-requirements minimal-reqs.txt pytest -s -n logical

cov:
	uv run --extra ci --isolated pytest -n logical -s --cov=kfactory --cov-branch --cov-report=xml

dev-cov:
	uv run --extra ci --isolated pytest -n logical -s --cov=kfactory --cov-report=term-missing:skip-covered

venv:
	uv venv -p 3.13

lint:
	uv run ruff check .

mypy:
	uv run dmypy run src/kfactory

pylint:
	pylint kfactory

pydocstyle:
	pydocstyle kfactory

doc8:
	doc8 docs/

autopep8:
	autopep8 --in-place --aggressive --aggressive **/*.py

codestyle:
	pycodestyle --max-line-length=88

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

update-pre:
	pre-commit autoupdate --bleeding-edge

release:
	tbump ${VERSION}
release-dr:
	tbump --dry-run ${VERSION}

gds-upload:
	gh release upload v0.6.0 gds/gds_ref/*.gds --clobber

gds-download:
	gh release download v0.6.0 -D gds/gds_ref/ --clobber

.PHONY: build docs test test-min
