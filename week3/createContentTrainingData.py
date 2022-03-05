# %%
import fileinput
import string
from types import SimpleNamespace
from typing import Union
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import click
from nltk.stem.snowball import SnowballStemmer
import nltk


# %%
cfg = SimpleNamespace()
cfg.input = Path("/workspace/search_with_machine_learning_course/data/pruned_products/")
cfg.output_dir = Path("/workspace/datasets/fasttext/")
cfg.pcats_df_file = "pcats_df.pk"
cfg.train_df_file = "pcats_train_df.pk"
cfg.test_df_file = "pcats_test_df.pk"
cfg.random_seed = 42

# %%
# Used for debugging while running interactively
dbg = SimpleNamespace()

# %%
translation_table = str.maketrans("", "", string.punctuation)


def transform_name(name: str):
    name = name.lower()
    # remove punctuation
    name = name.translate(translation_table)
    # to tokens
    tokens = nltk.word_tokenize(name)
    # snowball stemmer
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


@click.group(context_settings={"show_default": True})
@click.option(
    "--input",
    default=cfg.input,
    help="The directory containing product data",
)
@click.option(
    "--output_dir",
    default=cfg.output_dir,
    help="The directory where output files are stored",
)
def cli(input, output_dir):
    cfg.input = Path(input)
    cfg.output_dir = Path(output_dir)


@cli.command()
def prepare_pcats_df():
    """Create pickled dataframe from products xml files.
    """
    pcats_xml_to_df(cfg.input, cfg.output_dir / cfg.pcats_df_file)


@cli.command()
@click.option(
    "--run_subdir",
    default=".",
    help="The subdirectory of output_dir where fasttext run files are stored",
)
@click.option(
    "--transform",
    default=True,
    help="Preprocess product names if true",
)
@click.option(
    "--min_products",
    default=0,
    type=int,
    help="The minimum number of products per category",
)
@click.option(
    "--max_depth",
    default=0,
    type=int,
    help="Use category ancestor at level max_depth (if max_depth > 0)",
)
def prepare_input(run_subdir, transform, min_products, max_depth):
    """
    Create fasttext train and test files.
    """
    write_fasttext_files(
        data_output_dir=cfg.output_dir / run_subdir,
        transform=transform,
        min_products=min_products,
        max_depth=max_depth,
    )


def df_to_text(df, filepath: Path, transform: bool):
    print(f"Writing {len(df)} rows to {filepath}")
    # df["label"] = "__label__" + df["cat"]
    with filepath.open("w") as f:
        for row in df.itertuples():
            text = transform_name(row.name) if transform else row.name
            f.write(f"__label__{row.cat} {text}\n")


# %%
def prune_categories_for_min_products(df, min_products: int):
    # panda series with index=cat and value=count
    counts = df.value_counts(subset=["cat"])
    cats_to_keep = counts[counts >= min_products].index.get_level_values(0)
    pruned_df = df[df["cat"].isin(cats_to_keep)]
    print(f"min_products = {min_products}: ", end="")
    print(
        f"keeping {len(cats_to_keep)}/{len(counts)} categories and {len(pruned_df)}/{len(df)} rows."
    )
    return pruned_df


# %%
# Return mapping from a category to ancestor at a given level
# For eg. for category G with path from root A -> B -> C -> D -> E -> F -> G
# calling with max_depth = 2 will return {"G": "B"} and with 3 will return {"G": "C"}
#
# Sample child in xml file:
# <category>
#   <id>abcat0010000</id>
#   <name>Gift Center</name>
#   <path>
#     <category>
#       <id>cat00000</id>
#       <name>Best Buy</name>
#     </category>
#     <category>
#       <id>abcat0010000</id>
#       <name>Gift Center</name>
#     </category>
#   </path>
# ...
# </category>
def get_pruned_category_tree_mappings(max_depth: int):
    filepath = "/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml"
    tree = ET.parse(filepath)
    root = tree.getroot()
    cat_map = {}
    for child in root:
        cat = child.findtext("./id")
        name = child.findtext("./name")
        e_path = child.find("./path")
        e_cats = e_path.findall("./category")
        path_cats = [e.findtext("./id") for e in e_cats]
        path_names = [e.findtext("./name") for e in e_cats]
        pruned_cat = path_cats[max_depth - 1] if len(path_cats) > max_depth else cat
        cat_map[cat] = {
            "cat": cat,
            "name": name,
            "path_cats": path_cats,
            "path_names": path_names,
            "pruned_cat": pruned_cat,
        }

    return cat_map


# %%
def map_categories_to_ancestor_at_depth(df, data_output_dir, max_depth: int = 0):
    df["ocat"] = df["cat"].copy()
    # map only if max_depth > 0
    if max_depth == 0:
        return df

    cat_map = get_pruned_category_tree_mappings(max_depth)
    pmap = {k: v["pruned_cat"] for k, v in cat_map.items()}
    df["cat"] = df["cat"].map(pmap)

    # Some categories have no mapping in category details file, so filter df
    no_cat_map_df = df[df["cat"].isnull()]
    no_cat_map_df.to_csv(data_output_dir / "missing_category_mapping.csv")
    df = df.dropna(subset=["cat"])

    print(
        f"max_depth={max_depth}: categories pruned from {df['ocat'].nunique()} to {df['cat'].nunique()}"
    )
    return df


# %%
def write_fasttext_files(
    data_output_dir: Path, transform: bool, min_products: bool, max_depth: int
):
    data_output_dir.mkdir(exist_ok=True)
    dbg.data_output_dir = data_output_dir

    fp = cfg.output_dir / cfg.pcats_df_file
    print(f"Reading df from {fp}")
    df = pd.read_pickle(str(fp))
    df = df.sample(frac=1, random_state=cfg.random_seed)
    # prune categories for min_products
    df = prune_categories_for_min_products(df, min_products)
    # map categories to ancestor
    if max_depth > 0:
        df = map_categories_to_ancestor_at_depth(
            df, data_output_dir=data_output_dir, max_depth=max_depth
        )
    train_df = df[:10000]
    test_df = df[-10000:]
    df_to_text(train_df, data_output_dir / "products.train", transform=transform)
    df_to_text(test_df, data_output_dir / "products.test", transform=transform)


# %%
def pcats_xml_to_df(input: Union[str, Path], pcats_df_file: Union[str, Path]):
    rows = []
    for filepath in Path(input).glob("*.xml*"):
        print("Processing %s" % filepath)
        f = fileinput.hook_compressed(filepath, "rb")
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            name = child.findtext("./name")
            e_cats = child.findall("./categoryPath/category")
            if name is None or len(e_cats) == 0:
                continue
            cat = e_cats[-1].findtext("./id")
            if cat is None:
                continue
            # Replace newline chars with spaces so fastText doesn't complain
            name = name.replace("\n", " ")
            row = {
                "name": name,
                "cat": cat,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"No. of rows = {len(df)}")
    print(f"Writing product categories dataframe to: {pcats_df_file}")
    df.to_pickle(str(pcats_df_file))


if __name__ == "__main__":
    cli()
