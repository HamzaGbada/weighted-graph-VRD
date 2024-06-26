import unittest

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.DataLoader.sroie_dataloader import SROIE
from src.utils.setup_logger import logger
import logging
import sys
from src.DataLoader.funsd_dataloader import FUNSD
from src.DataLoader.cord_dataloader import CORD


class TestFunsdDataLoader(unittest.TestCase):
    def test_json_linking(self):
        train_set = CORD(train=True, download=True)
        logger.debug(train_set[0])
        logger.debug(train_set.data[0])
        self.assertEqual(train_set.data[0][0], "0060036622.png")


class TestCordDataLoader(unittest.TestCase):
    def test_json(self):
        train_set = CORD(train=False, download=True)
        path = "../../data/cord_train/image/"
        image = Image.open(path + train_set.data[0][0])
        # convert image to numpy array
        image = np.asarray(image)
        bbox = train_set.data[0][1]["boxes"]
        title = train_set.data[0][1]["labels"][0]
        xmin, ymin, xmax, ymax = bbox[0]
        logger.debug(f"the bbox: {bbox[0]}")
        logger.debug(f"the xmin: {xmin}")
        logger.debug(f"the ymin: {ymin}")
        logger.debug(f"the xmax: {xmax}")
        logger.debug(f"the ymax: {ymax}")
        data = image[ymin:ymax, xmin:xmax]

        plt.imshow(data)
        # plt.imshow(image)
        plt.title(title)
        plt.show()

        self.assertEqual(train_set.data[0][0], "receipt_00425.png")


class TestSROIEDataLoader(unittest.TestCase):
    def test_sroie(self):
        train_set = SROIE(train=True)
        nbr_of_node = train_set.data[0][1]["boxes"].shape
        logger.debug(f"The shape of bbox in the first doc Dataset: \n{nbr_of_node}")
        logger.debug(
            f"The shape of bbox in the first doc Dataset: \n{len(train_set.data[0][1]['boxes'])}"
        )
        logger.debug(f"The bbox in the first doc Dataset: \n{train_set.data[0]}")
        # logger.debug(f"The bbox in the first doc Dataset: \n{train_set[55]}")
        plt.imshow(train_set[0][0].permute(1, 2, 0))
        plt.show()


if __name__ == "__main__":
    unittest.main()
