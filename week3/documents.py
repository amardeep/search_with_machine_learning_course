#
# A simple endpoint that can receive documents from an external source, mark them up and return them.  This can be useful
# for hooking in callback functions during indexing to do smarter things like classification
#
from flask import (
    Blueprint, request, abort, current_app, jsonify
)
import fasttext
import json
import nltk
from nltk.stem.snowball import SnowballStemmer
import string

bp = Blueprint('documents', __name__, url_prefix='/documents')

SYNONYMS_THRESHOLD_SCORE = .9

translation_table = str.maketrans("", "", "®©™" + string.punctuation)

def get_synonyms(syns_model, name: str) -> list[str]:
    name = name.strip().lower()
    name = name.translate(translation_table)
    tokens = nltk.word_tokenize(name)
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(t) for t in tokens]

    syns = set()
    for token in tokens:
        # Returns list of <score, term> pairs
        nns = syns_model.get_nearest_neighbors(token, k=20)
        nns_terms = [nn[1] for nn in nns if nn[0] > SYNONYMS_THRESHOLD_SCORE]
        syns.update(nns_terms)

    return list(syns)

# Take in a JSON document and return a JSON document
@bp.route('/annotate', methods=['POST'])
def annotate():
    if request.mimetype == 'application/json':
        the_doc = request.get_json()
        # print(the_doc)
        response = {}
        cat_model = current_app.config.get("cat_model", None) # see if we have a category model
        syns_model = current_app.config.get("syns_model", None) # see if we have a synonyms/analogies model
        # We have a map of fields to annotate.  Do POS, NER on each of them
        sku = the_doc["sku"]
        for item in the_doc:
            the_text = the_doc[item]
            if the_text is not None and the_text.find("%{") == -1:
                if item == "name":
                    if syns_model is not None:
                        syns = get_synonyms(syns_model=syns_model, name=the_text)
                        # print(f"Synonyms for {the_text}:\n", syns)
                        response["name_synonyms"] = syns
        return jsonify(response)
    abort(415)
