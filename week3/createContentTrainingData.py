# %%
import string
from types import SimpleNamespace
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import click
from nltk.stem.snowball import SnowballStemmer
import nltk

#####
# NOTE:
# - Logic to parse xml files is moved to createProductsDataframe.py. This file
#   reads pickled dataframe constructed in that file.
#####

# %%
cfg = SimpleNamespace()
cfg.output_dir = Path("/workspace/datasets/fasttext/")
cfg.products_df_file = Path("/workspace/datasets/fasttext/pruned_products_df.pk")
cfg.random_seed = 42

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
    "--products_df",
    default=cfg.products_df_file,
    help="Pickled dataframe with product data",
)
@click.option(
    "--output_dir",
    default=cfg.output_dir,
    help="The directory where output files are stored",
)
def cli(products_df, output_dir):
    cfg.products_df_file = Path(products_df)
    cfg.output_dir = Path(output_dir)


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
        products_df_file=cfg.products_df_file,
        data_output_dir=cfg.output_dir / run_subdir,
        transform=transform,
        min_products=min_products,
        max_depth=max_depth,
    )


def df_to_text(df, filepath: Path, transform: bool):
    print(f"Writing {len(df)} rows to {filepath}")
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
    products_df_file: Path,
    data_output_dir: Path,
    transform: bool,
    min_products: bool,
    max_depth: int,
):
    data_output_dir.mkdir(exist_ok=True)

    print(f"Reading df from {products_df_file}")
    df = pd.read_pickle(str(products_df_file))
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
if __name__ == "__main__":
    cli()

# %%
