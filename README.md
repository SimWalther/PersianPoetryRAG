# Persian Poetry RAG

## Environment Variables
For the `pipeline` and `rag_backend`, you need to create an environment file .env with the necessary configuration. You can see an example inside the `.env_example` files. It contains the variables of the postgres database, and if the huggingface models used request it, the huggingface token.

## Directories

- **pipeline**: in pipeline you find the DVC pipeline to reproduce the texts translation, chunking, embedding creation, and RAG evaluation
- **rag**: this folder contains the files shared between the pipeline and the rag_backend  
- **rag_backend**: rag backend is the backend of the web application with bentoml
- **rag_frontend**: rag frontend is the frontend of the web application with vuejs

It is necessary to run the pipeline in order to have the text embeddings in the database. 

Note that you need the data to run the pipeline.

### Python

This project uses Python 3.12, make sure to have it installed.

### Running the RAG

To run the RAG, you need to run:

1. the Postgres database
2. the backend
3. the frontend

### Postgres

You need to have a Postgres database running with the required data (masnavi, ghazal, ...) tables.

### Pipeline (DVC)

Go to the pipeline directory (`cd pipeline`) and activate the python environment with `source .venv/bin/activate`. Make sure you have the dependencies installed by doing `pip install -r requirements.txt`.

This project uses [DVC](https://dvc.org/), you can reproduce the experiment with `dvc repro`.

### Backend (BentoML)

Go to the rag_backend directory (`cd rag_backend`) and activate the python environment with `source .venv/bin/activate`.

Make sure you have the dependencies installed by doing `pip install -r requirements.txt`.

Finally, run the backend with `bentoml serve .`

### Frontend (VueJS)

Go to the rag_frontend directory (`cd rag_frontend`)

You need to have `npm` installed.

You can install the npm dependencies with `npm install`.

Then you run the frontend with `npm run dev`. It will tell you the local url where you can access the user interface and send queries to the RAG. 