import random
import argparse
from pathlib import Path
import pandas as pd

# Dataframe with products data from xml files
products_df_file = Path("/workspace/datasets/fasttext/pruned_products_df.pk")
output_file = Path("/workspace/datasets/fasttext/titles.txt")

parser = argparse.ArgumentParser(description="Extract product titles")

general = parser.add_argument_group("general")
general.add_argument(
    "--products_df",
    default=products_df_file,
    help="Dataframe with products data from xml files",
)
general.add_argument("--output", default=output_file, help="The file to output to")
# Consuming all of the product data takes a while. But we still want to be able to obtain a representative sample.
general.add_argument(
    "--sample_rate",
    default=0.1,
    type=float,
    help="The rate at which to sample input (default is 0.1)",
)

args = parser.parse_args()
products_df_file = Path(args.products_df)
output_file = Path(args.output)
output_file.parent.mkdir(exist_ok=True)
sample_rate = args.sample_rate


def transform_training_data(name):
    name = name.replace("\n", " ")


df: pd.DataFrame = pd.read_pickle(products_df_file)
df = df.sample(frac=sample_rate, random_state=42)

print("Writing results to %s" % output_file)
with open(output_file, "w") as output:
    for row in df.itertuples():
        text = transform_training_data(row.name)
        output.write(f"{text}\n")
