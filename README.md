# Structured Document Graph

## Graph builder
To build a graph-based dataset check the following command:
```shell script
    $ python graph_builder.py -h   
        usage: graph_builder.py [-h] [-d DATASET] [-t TRAIN]
   
        This command creates a graph based dataset for node classification for a specific dataset in order to extract entites from Visually Rich Document. The default
        is: "./data/<DATASET_NAME>/<Train||Test>/"

        optional arguments:
          -h, --help            show this help message and exit
          -d DATASET, --dataset DATASET
                                Choose the dataset to use. It can be "FUNSD" or "CORD"
          -t TRAIN, --train TRAIN
                                Boolean to choose between train or test dataset

```