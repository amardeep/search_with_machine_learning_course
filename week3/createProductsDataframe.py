# %%
import fileinput
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import click


# %%
# Reads xml files in input_dir
# Generates in <output_dir>
# - <output_prefix>_df.pk - pickled dataframe with products and categories
# - <output_prefix>.csv - csv with products and categories
def create_products_df(input_dir: Path, output_dir: Path, output_prefix: str):
    rows = []
    for filepath in Path(input_dir).glob("*.xml*"):
        print("Processing %s" % filepath)
        f = fileinput.hook_compressed(filepath, "rb")
        root = ET.parse(f).getroot()
        for child in root:
            name = child.findtext("./name")
            if not name:
                continue
            e_cats = child.findall("./categoryPath/category")
            cats = [e.findtext("./id") for e in e_cats]
            cat_names = [e.findtext("./name") for e in e_cats]

            # Replace newline chars with spaces so fastText doesn't complain
            name = name.replace("\n", " ")

            row = {
                "name": name,
                "cat": cats[-1] if cats else None,
                "cat_name": cat_names[-1] if cat_names else None,
                "cats": cats,
                "cat_names": cat_names,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"No. of rows = {len(df)}")

    df_fp = output_dir / f"{output_prefix}_df.pk"
    csv_fp = output_dir / f"{output_prefix}.csv"

    print(f"Writing product categories dataframe to: {df_fp}")
    df.to_pickle(str(df_fp))

    print(f"Writing product categories csv to: {df_fp}")
    df.to_csv(str(csv_fp), columns=["cat", "cat_name", "name"], index=False)

    return df


# %%
@click.group()
def cli():
    "Create pickled dataframes from products xml files"
    pass


@cli.command()
def pruned_products():
    "Create dataframe for pruned products"
    input_dir = Path(
        "/workspace/search_with_machine_learning_course/data/pruned_products/"
    )
    output_dir = Path("/workspace/datasets/fasttext/")
    output_prefix = "pruned_products"
    prunded_df = create_products_df(input_dir, output_dir, output_prefix)


@cli.command()
def phone_products():
    "Create dataframe for phone products"
    input_dir = Path(
        "/workspace/search_with_machine_learning_course/data/phone_products/"
    )
    output_dir = Path("/workspace/datasets/fasttext/")
    output_prefix = "phone_products"
    phone_df = create_products_df(input_dir, output_dir, output_prefix)


# %%
if __name__ == "__main__":
    cli()
