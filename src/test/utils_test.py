import unittest

from src.DataLoader.funsd_dataloader import FUNSD
from src.utils.utils import funsd_to_CSV, cord_to_CSV, sroie_to_CSV, feature_embedding, edge_metric_calculation, \
    edge_builder
from src.utils.setup_logger import logger


class TestUtils(unittest.TestCase):

    def test_funsd_to_csv(self):
        funsd_to_CSV()

    def test_cord_to_csv(self):
        cord_to_CSV()

    def test_sroie_to_csv(self):
        sroie_to_CSV()

    def test_graph_to_csv(self):
        train_set = FUNSD(train=False, download=True)
        logger.debug(f"the whole dataset length: \n {len(train_set.data)}")
        logger.debug(f"the whole embedded dataset length: \n {len(feature_embedding(train_set))}")

    def test_edge_metric(self):
        doc_index = 0
        train_set = FUNSD(train=False, download=True)
        # TODO: Bug found is in embedded dataset calculation or edge array
        embedded_dataset = feature_embedding(train_set)
        # node_feature = embedded_dataset[doc_index]["boxes_features"]
        # edge_array = edge_builder(embedded_dataset)
        # logger.debug(f"edge array: {edge_array[doc_index]}")
        # logger.debug(f"the len of edge array (first one): {len(edge_array[doc_index])}")
        # logger.debug(f"the size of edge array (should be equal to nbr of doc): {len(edge_array)}")
        # data_per_doc = train_set.data[doc_index]
        # edge_metric_dict = edge_metric_calculation(edge_array[doc_index], node_feature, data_per_doc)
        # logger.debug(f"the edge_metric_dict: \n {edge_metric_dict}")


if __name__ == '__main__':
    unittest.main()
