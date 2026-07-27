#!/bin/sh

set -eu

echo "Preparing Research Paper Intelligence runtime data."

# This command downloads the dataset and processes the data.
preprocess-papers

# These commands download existing Hugging Face artefacts or generate them
# when remote artefacts are unavailable.
generate-embeddings
generate-faiss-index
generate-tfidf-index

echo "Runtime data preparation completed successfully."