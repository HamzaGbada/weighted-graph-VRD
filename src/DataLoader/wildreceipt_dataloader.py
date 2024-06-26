import json
import os
from typing import Any, List, Tuple, Dict

import numpy as np
from doctr.datasets.datasets import VisionDataset
from doctr.datasets.utils import convert_target_to_relative

from src.utils.setup_logger import logger
from src.utils.utils import convert_xmin_ymin, get_area


class WILDRECEIPT(VisionDataset):
    """
    :return:
        Bounding boxes are in the Format (xmin, ymin, xmax, ymax) top left, bottom right corners
    """
    dataset = ('https://download.openmmlab.com/mmocr/data/wildreceipt.tar', 'wildreceipt.tar')

    def __init__(self, train: bool = True, **kwargs: Any) -> None:
        url, filename = self.dataset
        super().__init__(url, filename, None, True, pre_transforms=convert_target_to_relative, **kwargs)

        tmp_root = os.path.join(self.root, 'wildreceipt/')
        self.train = train

        self.data: List[Tuple[str, Dict[str, Any]]] = []

        self.filename = "train.txt" if self.train else "test.txt"
        file_path = os.path.join(tmp_root, self.filename)
        # logger.debug(f'the file names: {tmp_root}')
        with open(file_path, 'r') as file:
            data = file.read()
        # Split the text file into separate JSON strings
        self.files = []
        json_strings = data.strip().split('\n')
        for json_string in json_strings:
            json_data = json.loads(json_string)
            file_name = json_data['file_name']
            self.files.append(file_name)
            annotations = json_data['annotations']
            _targets = [(convert_xmin_ymin(annotation['box']), annotation['text'].lower(), annotation['label'])
                        for annotation in annotations if get_area(convert_xmin_ymin(annotation['box'])) >= 50]
            if _targets:
                box_targets, text_units, labels = zip(*_targets)

                self.data.append((
                    file_name,
                    dict(boxes=np.asarray(box_targets, dtype=int), labels=list(labels),
                         text_units=list(text_units)),
                ))
        self.root = tmp_root


    def extra_repr(self) -> str:
        return f"train={self.train}"
