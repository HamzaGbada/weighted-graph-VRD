import os

from dgl import batch
from dgl.data import DGLDataset, load_graphs
from torch import zeros, bool

from src.utils.setup_logger import logger


class DocumentGraphDataset(DGLDataset):
    def __init__(self, data_name: str, path: str):
        if data_name == "FUNSD":
            logger.debug(f"the dataset name {data_name}")
            path_test = path + "FUNSD/test/"
            path_train = path + "FUNSD/train/"
            self.num_classes = 4
        elif data_name == "CORD":
            path_test = path + "CORD/test/"
            path_train = path + "CORD/train/"
            self.num_classes = 30
        elif data_name == "SROIE":
            path_test = path + "SROIE/test/"
            path_train = path + "SROIE/train/"
            self.num_classes = 5

        else:
            logger.debug(f"Put a valide dataset name!")
            return

        # giving file extension
        ext = "bin"

        graph_list_train = [
            load_graphs(path_train + files)[0][0]
            for files in os.listdir(path_train)
            if files.endswith(ext)
        ]
        graph_list_test = [
            load_graphs(path_test + files)[0][0]
            for files in os.listdir(path_test)
            if files.endswith(ext)
        ]

        self.graph_train = batch(graph_list_train)
        self.graph_test = batch(graph_list_test)

        self.graph_train = batch([self.graph_train, self.graph_test])
        logger.debug(self.graph_train.number_of_nodes())
        logger.debug(self.graph_test.number_of_nodes())
        super().__init__(name="document_graph")

    def process(self):
        #
        n_nodes = self.graph_train.number_of_nodes()
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = zeros(n_nodes, dtype=bool)
        val_mask = zeros(n_nodes, dtype=bool)
        test_mask = zeros(n_nodes, dtype=bool)

        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def __getitem__(self, train: bool):
        if train:
            return self.graph_train
        return self.graph_test

    def __len__(self):
        return 1
