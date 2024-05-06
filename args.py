import argparse

# Argument parsing
parser = argparse.ArgumentParser(
    description="This command creates a graph based dataset for node classification for "
    "a specific dataset in order to extract entites from Visually Rich "
    'Document. The default is: "./data/<DATASET_NAME>/<Train||Test>/"'
)
parser.add_argument(
    "-d",
    "--dataset",
    help='Choose the dataset to use. It can be "FUNSD", "CORD" or "SROIE"',
    default="FUNSD",
)
parser.add_argument(
    "-t",
    "--train",
    help="Boolean to choose between train or test dataset",
    default=False,
)

argument = parser.parse_args()


parser1 = argparse.ArgumentParser()

parser1.add_argument(
    "-d",
    "--dataname",
    type=str,
    default="FUNSD",
    choices=["FUNSD", "SROIE", "CORD"],
    help="Selecting the dataset for your model's training.",
)

parser1.add_argument(
    "-p",
    "--path",
    type=str,
    default="data/",
    help="Selecting the dataset path for the model's training.",
)

parser1.add_argument(
    "-hs", "--hidden_size", type=int, default=16, help="GCN hidden size."
)

parser1.add_argument(
    "-hl", "--hidden_layers", type=int, default=10, help="Number of GCN hidden Layers."
)

parser1.add_argument(
    "-lr", "--learning_rate", type=float, default=0.01, help="The learning rate."
)

parser1.add_argument(
    "-e", "--epochs", type=int, default=50, help="The number of epochs."
)

argument1 = parser1.parse_args()
