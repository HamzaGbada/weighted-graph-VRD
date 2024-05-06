from math import sqrt

import numpy as np
from torch import from_numpy


class SpatialEmbedding:
    """
        Spatial Embedding for all bbox in one document
        :return: List
            [
                torch.tensor(bbox1)=[xmin, ymin, xmax, ymax, x_center, y_center,width, height, euclidean_distance]
                torch.tensor(bbox2)=[xmin, ymin, xmax, ymax, x_center, y_center,width, height, euclidean_distance]
                ...
                torch.tensor(bboxN)=[xmin, ymin, xmax, ymax, x_center, y_center,,width, height, euclidean_distance]
            ]
    """

    def __init__(self, bbox: np.ndarray, document: np.ndarray) -> None:
        super().__init__()
        img_c = (document.shape[0] // 2, document.shape[1] // 2)
        self.embedding = []
        for b in bbox:
            xc = (b[0] + b[2]) / 2
            yc = (b[1] + b[3]) / 2
            xc_1, yc_2 = xc-img_c[0] ,yc-img_c[1]
            rho1, theta1 = self.cart2pol(xc, yc)
            rho2, theta2 = self.cart2pol(xc_1, yc_2)
            w = b[2] - b[0]
            h = b[3] - b[1]
            dist = sqrt((img_c[0] - xc) ** 2 + (img_c[1] - yc) ** 2)
            self.embedding.append(from_numpy(np.append(b, [xc, yc, w, h, dist, rho1, theta1, rho2, theta2]+[0] * 487)))

    def __getitem__(self, index: int):
        return self.embedding[index]

    @staticmethod
    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi
