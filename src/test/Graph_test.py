import os
import unittest

import matplotlib.pyplot as plt
import networkx as nx
from dgl import to_networkx, load_graphs, to_simple, add_reverse_edges, batch
from torch import zeros

from src.DataLoader.cord_dataloader import CORD
from src.DataLoader.funsd_dataloader import FUNSD
from src.DataLoader.graph_dataloader import DocumentGraphDataset
from src.GraphModule.GraphConstructor import GraphConstructor
from src.utils.setup_logger import logger


class TestGraph(unittest.TestCase):
    def test_graph_constructor(self):
        train_set = FUNSD(train=True, download=True)

        G_list = GraphConstructor(train_set).graph_set
        logger.debug(f"the nbr of graphs initial: {len(G_list)}")
        logger.debug(f"nbr of document: {len(train_set.data)}")
        logger.debug(f"nbr of node in the first document: {G_list[0].number_of_nodes()}")
        logger.debug(f"The number of bbox in the first document: {train_set.data[0][1]['boxes'].shape}")
        G = to_networkx(G_list[0])
        nx.draw(G)
        plt.show()

    def test_graph_load(self):
        path_test = '../../data/FUNSD/test/'
        path_train = '../../data/FUNSD/train/'

        ext = 'bin'

        graph_list = [load_graphs(path_test+files)[0][0] for files in os.listdir(path_test) if files.endswith(ext)]
        len_glist = [x.ndata['label'].shape[0] for x in graph_list]
        nbr_nodes = [x.number_of_nodes() for x in graph_list]
        batched = batch(graph_list)
        logger.debug(f"number of node after {batched.number_of_nodes()}")
        graph = graph_list[0]
        n_nodes = graph.number_of_nodes()

        n_nodes = graph_list[0].number_of_nodes()
        n_train = int(n_nodes * 0.8)
        train_mask = zeros(n_nodes, dtype=bool)
        val_mask = zeros(n_nodes, dtype=bool)

        test_mask = zeros(n_nodes, dtype=bool)
        train_mask[:n_train] = True
        val_mask[n_train:] = True
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        logger.debug(f"train_mask {train_mask}")
        logger.debug(f"val_mask {val_mask}")
        logger.debug(f"test_mask {test_mask}")

        logger.debug(f"the graph {graph}")
        G = to_networkx(graph_list[0])
        nx.draw(G)
        plt.show()

    def test_graph_dataloader(self):
        dataset = DocumentGraphDataset("FUNSD")
        graph_test = dataset[True]
        logger.debug(f"number of classes {dataset.num_classes}")
        G = to_networkx(graph_test)
        nx.draw(G)
        plt.show()


