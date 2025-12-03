# BentoML RAG backend

## Serving localy

You can serve the RAG locally with `bentoml serve .`

## Create docker container

First, build with: `bentoml build`

Then, you can create a docker container with: `bentoml containerize rag_service:latest`

After having created it, it will output how to run this docker container for instance: `docker run --rm -p 3000:3000 rag_service:VERSION`, where version is the bentoml tag of the current RAG.

