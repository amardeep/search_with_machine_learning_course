#
# The main search hooks for the Search Flask application.
#
from flask import Blueprint, redirect, render_template, request, url_for
import rich

from week1.opensearch import get_opensearch

bp = Blueprint("search", __name__, url_prefix="/search")


# Process the filters requested by the user and return a tuple that is appropriate for use in:
#  the query, URLs displaying the filter and the display of the applied filters
# filters -- convert the URL GET structure into an OpenSearch filter query
# display_filters -- return an array of filters that are applied that is appropriate for display
# applied_filters -- return a String that is appropriate for inclusion in a URL as part of a query string.  This is basically the same as the input query string
def process_filters(filters_input):
    # Filters look like: &filter.name=regularPrice&regularPrice.key={{ agg.key }}&regularPrice.from={{ agg.from }}&regularPrice.to={{ agg.to }}
    filters = [] # filter entries in bool query
    must_not = [] # must_not entries in bool query
    display_filters = [] # strings to display in UI for current filters
    applied_filters = "" # partial query string for filters

    for filter in filters_input:
        type = request.args.get(f"{filter}.type")
        display_name = request.args.get(f"{filter}.displayName", filter)
        filter_from = request.args.get(f"{filter}.from")
        filter_to = request.args.get(f"{filter}.to")
        key = request.args.get(f"{filter}.key")

        # We need to capture and return what filters are already applied so they can be automatically added to any existing links we display in aggregations.jinja2
        applied_filters += f"&filter.name={filter}&{filter}.type={type}&{filter}.displayName={display_name}"
        display_filter = f"{display_name}"
        if key is not None:
            applied_filters += f"&{filter}.key={key}"
            display_filter += f": {key}"
        if filter_from is not None:
            applied_filters += f"&{filter}.from={filter_from}"
            display_filter += f" from: {filter_from}"
        if filter_to is not None:
            applied_filters += f"&{filter}.to={filter_to}"
            display_filter += f" to: {filter_to}"

        display_filters.append(display_filter)

        # filters get used in create_query below.  display_filters gets used by display_filters.jinja2 and applied_filters gets used by aggregations.jinja2 (and any other links that would execute a search.)
        if type == "range":
            o = {}
            if filter_from:
                o["gte"] = filter_from
            if filter_to:
                o["lt"] = filter_to
            filters.append({"range": {filter: o}})
        elif type == "terms":
            filters.append({"term": {f"{filter}.keyword": key}})
        elif type == "missing":
            must_not.append({"exists": {"field": key}})

    return filters, must_not, display_filters, applied_filters


# Our main query route.  Accepts POST (via the Search box) and GETs via the clicks on aggregations/facets
@bp.route("/query", methods=["GET", "POST"])
def query():
    opensearch = get_opensearch()
    # Put in your code to query opensearch.  Set error as appropriate.
    error = None
    user_query = None
    query_obj = None
    display_filters = None
    applied_filters = ""
    filters = None
    sort = "_score"
    sortDir = "desc"
    if request.method == "POST":  # a query has been submitted
        user_query = request.form["query"]
        if not user_query:
            user_query = "*"
        sort = request.form["sort"]
        if not sort:
            sort = "_score"
        sortDir = request.form["sortDir"]
        if not sortDir:
            sortDir = "desc"
        query_obj = create_query(user_query, [], [], sort, sortDir)
    elif request.method == "GET":
        # Handle the case where there is no query or just loading the page
        user_query = request.args.get("query", "*")
        filters_input = request.args.getlist("filter.name")
        sort = request.args.get("sort", sort)
        sortDir = request.args.get("sortDir", sortDir)
        if filters_input:
            (filters, must_not, display_filters, applied_filters) = process_filters(
                filters_input
            )
            query_obj = create_query(user_query, filters, must_not, sort, sortDir)
        else:
            query_obj = create_query(user_query, [], [], sort, sortDir)

    print("Query obj:")
    rich.print(query_obj)
    response = opensearch.search(body=query_obj, index="bbuy_products1")
    # Postprocess results here if you so desire

    # print(response)
    if error is None:
        return render_template(
            "search_results.jinja2",
            query=user_query,
            search_response=response,
            display_filters=display_filters,
            applied_filters=applied_filters,
            sort=sort,
            sortDir=sortDir,
        )
    else:
        redirect(url_for("index"))


def create_query(user_query, filters, must_not, sort="_score", sortDir="desc"):
    print()
    print(f"User query: {user_query}")
    print(f"Filters: {filters}")
    print(f"Must nots: {must_not}")
    print(f"Sort: {sort} {sortDir}")
    print()

    # Query for which documents to score
    match_query = {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": user_query,
                        "fields": [
                            "name^100",
                            "shortDescription^50",
                            "longDescription^10",
                            "department",
                        ],
                    }
                }
            ],
            "filter": filters,
            "must_not": must_not,
        }
    }

    aggs = {
        "department": {"terms": {"field": "department.keyword"}},
        "missing_images": {"missing": {"field": "image.keyword"}},
        "regularPrice": {
            "range": {
                "field": "regularPrice",
                "ranges": [
                    {
                        "key": "$",
                        "from": 0,
                        "to": 100,
                    },
                    {
                        "key": "$$",
                        "from": 100,
                        "to": 200,
                    },
                    {
                        "key": "$$$",
                        "from": 200,
                        "to": 300,
                    },
                    {
                        "key": "$$$$",
                        "from": 300,
                        "to": 400,
                    },
                    {
                        "key": "$$$$$",
                        "from": 400,
                    },
                ],
            }
        },
    }

    function_score_query = {
        "function_score": {
            "query": match_query,
            "boost_mode": "multiply",
            "score_mode": "avg",
            "functions": [
                {
                    "field_value_factor": {
                        "field": "salesRankShortTerm",
                        "modifier": "reciprocal",
                        "missing": 100000000,
                    }
                },
                {
                    "field_value_factor": {
                        "field": "salesRankMediumTerm",
                        "modifier": "reciprocal",
                        "missing": 100000000,
                    }
                },
                {
                    "field_value_factor": {
                        "field": "salesRankLongTerm",
                        "modifier": "reciprocal",
                        "missing": 100000000,
                    }
                },
            ],
        }
    }

    query_obj = {
        "size": 10,
        "query": function_score_query,
        "aggs": aggs,
    }
    return query_obj
