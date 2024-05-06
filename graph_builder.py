import networkx as nx
from dgl import to_networkx
import matplotlib.pyplot as plt

from args import argument
from src.DataLoader.cord_dataloader import CORD
from src.DataLoader.funsd_dataloader import FUNSD
from src.DataLoader.sroie_dataloader import SROIE
from src.GraphModule.GraphConstructor import GraphConstructor
from src.utils.setup_logger import logger

if __name__ == "__main__":
    # dataset_dict = {
    #     "FUNSD": FUNSD(train=argument.train, download=True),
    #     "CORD": CORD(train=argument.train, download=True)
    # }
    if argument.dataset == "FUNSD":
        train_set = FUNSD(train=argument.train == "True", download=True)
    if argument.dataset == "CORD":
        train_set = CORD(train=argument.train == "True", download=True)
    if argument.dataset == "SROIE":
        train_set = SROIE(train=argument.train == "True")

    # train_set = FUNSD(train=False, download=True)
    # This is the labels of each text unit in the first document
    # L = [label[1] for label in train_set.data[0][1]['labels']]
    # logger.debug(f"all labels {L}")
    # nbr_of_bbox = train_set.data[0][1]['boxes'].shape
    # title = train_set.data[0][1]['labels'][0][0]
    # logger.debug(f"word in the bbox : {title}")
    # logger.debug(f"The shape of bbox in the first doc Dataset: \n{nbr_of_bbox}")
    # logger.debug(f"The shape of bbox in the first doc Dataset: \n{len(train_set.data[0][1]['boxes'])}")
    # logger.debug(f"The bbox in the first doc Dataset: \n{train_set.data[0][1]['boxes']}")
    # logger.debug(f"class name : {type(train_set).__name__}")
    G_list = GraphConstructor(train_set).graph_set
    logger.debug(f"the nbr of graphs initial: {len(G_list)}")
    logger.debug(f"nbr of document: {len(train_set.data)}")
    logger.debug(f"nbr of node in the first document: {G_list[0].number_of_nodes()}")
    logger.debug(
        f"The number of bbox in the first document: {train_set.data[0][1]['boxes'].shape}"
    )
    # the nbr of edge = 37 * 38 / 2 (check the adjancy matrix tho see why)
    # G = to_networkx(G_list[0])
    # nx.draw(G)
    # plt.show()
