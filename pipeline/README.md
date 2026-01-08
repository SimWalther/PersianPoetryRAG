# Persian Poetry RAG

## Installing

### Python

This project uses Python 3.12, make sure to have it installed.

### Postgres

Make sure to have a Postgres database running with the required data (masnavi, ghazal, ...) tables and the pgvector extension activate (see: https://github.com/pgvector/pgvector)

### Activate environment

In the terminal inside the root directory do: `source .venv/bin/activate`

### Install dependencies

You can install python dependencies with: `pip install -r requirements.txt`

## Reproduce the experiment

This project uses [DVC](https://dvc.org/), you can reproduce the experiment with `dvc repro`.