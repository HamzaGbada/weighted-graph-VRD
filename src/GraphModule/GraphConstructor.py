from dgl import graph, add_reverse_edges
from dgl.data import save_graphs
from torch import tensor

from src.utils.setup_logger import logger
from src.utils.utils import feature_embedding, edge_builder, edge_metric_calculation


class GraphConstructor:
    def __init__(self, dataloader_set) -> None:
        self.graph_set = []
        embedded_dataset = feature_embedding(dataloader_set)
        logger.debug(f"SROIE data loader Check {embedded_dataset}")
        logger.debug(f"the labels: {dataloader_set.data[0][1]['labels']}")
        edge_array = edge_builder(embedded_dataset)
        for doc_index in range(len(embedded_dataset)):
            doc_name = dataloader_set.data[doc_index][0]

            doc_shape = (
                dataloader_set[doc_index][0].shape[1],
                dataloader_set[doc_index][0].shape[2],
            )

            # The node features
            node_feature = embedded_dataset[doc_index]["boxes_features"]
            top_dict = edge_metric_calculation(
                edge_array[doc_index], node_feature, doc_shape
            )
            # The edge configuration
            src = [node_index[0] for node_index in list(top_dict)]
            dst = [node_index[1] for node_index in list(top_dict)]

            src_ids = tensor(src)
            dst_ids = tensor(dst)
            logger.debug(f"source node list {src_ids}")
            logger.debug(f"dst node list {src_ids}")
            # The label of each node
            node_label = embedded_dataset[doc_index]["labels_encoded"]
            # edge feature
            edge_features = list(top_dict.values())

            # the graph building
            logger.debug(f"the number of node (by node label:  {len(node_label)}")
            logger.debug(f"the shape of node features: {node_feature.shape}")
            gph = graph((src_ids, dst_ids), num_nodes=len(node_label))
            logger.debug(
                f"The number of edge of to_bidirected graph {gph.number_of_edges()}"
            )

            logger.debug(f"the label list: {node_label}")
            gph.ndata["feat"] = node_feature
            gph.ndata["label"] = node_label
            gph.edata["weight"] = tensor(edge_features)
            # logger.debug(f" the connection of the current  node: {5} are: {gph.successors(5)}")
            gph = add_reverse_edges(gph, copy_ndata=True, copy_edata=True)
            # logger.debug(f" the connection of the current  node: {4} are: {gph.successors(4)}")

            isTrain = dataloader_set.train
            data_name = type(dataloader_set).__name__
            graph_path = (
                "data/"
                + data_name
                + "/train/"
                + doc_name.split(".", 1)[0]
                + "_graph.bin"
            )
            if not isTrain:
                graph_path = (
                    "data/"
                    + data_name
                    + "/test/"
                    + doc_name.split(".", 1)[0]
                    + "_graph.bin"
                )
            save_graphs(graph_path, gph)
            logger.debug(f"file is save to {graph_path}")
            self.graph_set.append(gph)

    def __getitem__(self, item: int):
        return self.graph_set[item]
