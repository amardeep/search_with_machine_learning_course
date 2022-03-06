#!/usr/bin/env bash
set -x 

export SYNONYMS_MODEL_LOC=/workspace/datasets/fasttext/phone_title_model.e25.mc5.bin
export FLASK_APP=week3
export FLASK_DEV=development

flask run