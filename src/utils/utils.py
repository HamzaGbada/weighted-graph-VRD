import logging
import math
import sys

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from scipy.spatial.distance import euclidean, chebyshev, cityblock, correlation, cosine
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from torch import swapaxes, from_numpy, numel, stack, flatten, tensor, int16
from torch.nn.functional import pad

from src.DataLoader.cord_dataloader import CORD
from src.DataLoader.funsd_dataloader import FUNSD
from src.DataLoader.sroie_dataloader import SROIE
from src.Encoder.image_embedding import SimpleCNNEncoder
from src.Encoder.spatial_embedding import SpatialEmbedding
from src.Encoder.word_embedding import Doc2VecEncoder
from src.utils.setup_logger import logger


# TODO: May be these method to_CSV are useless
def funsd_to_CSV():
    train_set = FUNSD(train=True, download=True)

    text_funsd = {
        "text_units": []
    }
    for k in range(len(train_set.data)):
        data_links = {}

        for links in train_set.data[k][1]['links']:
            if links:
                s = set()
                key = links[0][0]
                for link in links:
                    s.add(link[1])
                if key in data_links:
                    data_links[key] = data_links[key] | s
                else:
                    data_links[key] = s
        for i in data_links:
            ch = train_set.data[k][1]['text_units'][i] + " "
            for j in list(data_links[i]):
                ch += train_set.data[k][1]['text_units'][j] + " "
            text_funsd['text_units'].append(ch)

    pd.DataFrame(text_funsd).to_csv("../../data/funsd_text.csv")


def cord_to_CSV():
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)
    train_set = CORD(train=True, download=True)

    text_cord = {'text_units': [unit for unit_list in train_set.text_units
                                for unit in list(unit_list.values())]}
    pd.DataFrame(text_cord).to_csv("../../data/cord_text.csv")


def sroie_to_CSV():
    train_set = SROIE(train=True, download=True)
    text_sroie = {'text_units': [unit for unit_list in train_set
                                 for unit in unit_list[1]['labels']]}
    logger.debug(f"The sroie target {text_sroie['text_units']}")
    pd.DataFrame(text_sroie).to_csv("../../data/sroie_text.csv")


def feature_embedding(dataloader_set):
    """
    This function is for creating an embedded dataset

    :return:
        [
1st Document ->{
            'boxes_features': [
                torch.tensor(spatial_tensor_box1, image_tensor_box1, word_tensor_box1),
                torch.tensor(spatial_tensor_box2, image_tensor_box2, word_tensor_box2),
                torch.tensor(spatial_tensor_box3, image_tensor_box3, word_tensor_box3),
                .
                .
                .
                torch.tensor(spatial_tensor_boxN, image_tensor_boxN, word_tensor_boxN]
                ]
            'labels_encoded': [
                label_box1(int),
                label_box2(int),
                .
                .
                .
                label_boxN(int)
                ]
            }
2nd Document ->{
            'boxes_features': [
                torch.tensor(spatial_tensor_box1, image_tensor_box1, word_tensor_box1),
                torch.tensor(spatial_tensor_box2, image_tensor_box2, word_tensor_box2),
                torch.tensor(spatial_tensor_box3, image_tensor_box3, word_tensor_box3),
                .
                .
                .
                torch.tensor(spatial_tensor_boxN, image_tensor_boxN, word_tensor_boxN]
                ]
            'labels_encoded': [
                label_box1(int),
                label_box2(int),
                .
                .
                .
                label_boxN(int)
                ]
            }
kth Document ->{
            'boxes_features': [
                torch.tensor(spatial_tensor_box1, image_tensor_box1, word_tensor_box1),
                torch.tensor(spatial_tensor_box2, image_tensor_box2, word_tensor_box2),
                torch.tensor(spatial_tensor_box3, image_tensor_box3, word_tensor_box3),
                .
                .
                .
                torch.tensor(spatial_tensor_boxN, image_tensor_boxN, word_tensor_boxN]
                ]
            'labels_encoded': [
                label_box1(int),
                label_box2(int),
                .
                .
                .
                label_boxN(int)
                ]
            }
        ]

    """
    out_dim = 500
    # the number of channel here is the same as in the input of the first conv layer

    batch_size = 1
    logger.debug(f"feature Embedding start")
    home = str(Path.home())
    if dataloader_set.train:
        if type(dataloader_set).__name__ == "CORD":
            path = home + '/.cache/doctr/datasets/cord_train/image/'
        elif type(dataloader_set).__name__ == "FUNSD":
            path = home + '/.cache/doctr/datasets/funsd/dataset/training_data/images/'
        elif type(dataloader_set).__name__ == "SROIE":
            path = "data/SROIE2019/train/img/"
        else:
            logger.error(f"dataset is unkown!")
            return
    else:
        if type(dataloader_set).__name__ == "CORD":
            path = home + '/.cache/doctr/datasets/cord_test/image/'
        elif type(dataloader_set).__name__ == "FUNSD":
            path = home + '/.cache/doctr/datasets/funsd/dataset/testing_data/images/'
        elif type(dataloader_set).__name__ == "SROIE":
            path = "data/SROIE2019/test/img/"
        else:
            logger.error(f"dataset is unkown!")
            return
    if type(dataloader_set).__name__ == "CORD":
        encoded_dic = {'menu.sub_cnt': 0,
                       'sub_total.othersvc_price': 1,
                       'total.total_price': 2,
                       'menu.etc': 3,
                       'sub_total.discount_price': 4,
                       'menu.unitprice': 5,
                       'menu.discountprice': 6,
                       'void_menu.price': 7,
                       'menu.nm': 8,
                       'total.menutype_cnt': 9,
                       'sub_total.subtotal_price': 10,
                       'menu.sub_nm': 11,
                       'void_menu.nm': 12,
                       'menu.sub_unitprice': 13,
                       'menu.sub_etc': 14,
                       'menu.cnt': 15,
                       'menu.vatyn': 16,
                       'total.total_etc': 17,
                       'total.menuqty_cnt': 18,
                       'total.cashprice': 19,
                       'menu.num': 20,
                       'total.changeprice': 21,
                       'sub_total.tax_price': 22,
                       'sub_total.etc': 23,
                       'menu.price': 24,
                       'total.creditcardprice': 25,
                       'total.emoneyprice': 26,
                       'sub_total.service_price': 27,
                       'menu.itemsubtotal': 28,
                       'menu.sub_price': 29
                       }
    elif type(dataloader_set).__name__ == "FUNSD":
        encoded_dic = {'question': 0,
                       'answer': 1,
                       'header': 2,
                       'other': 3
                       }
    elif type(dataloader_set).__name__ == "SROIE":
        encoded_dic = {"TOTAL": 0,
                       "DATE": 1,
                       "ADDRESS": 2,
                       "COMPANY": 3,
                       "O": 4
                       }
    else:
        logger.error(f"dataset is unkown!")
        return
    word_embeder = Doc2VecEncoder(dataloader_set=dataloader_set, out_dim=out_dim)
    nbr_document = len(dataloader_set)


    data_encoded = []
    # TODO: Try to save the data encoded to npz files
    for doc_index in range(nbr_document):
        logger.debug(f"the docuemnt number processing now {doc_index}")

        bbox = dataloader_set.data[doc_index][1]['boxes']

        image_name = dataloader_set.data[doc_index][0]
        image = Image.open(path + image_name)
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = image.reshape((image.shape[0], image.shape[1], 1))
        nbr_of_channel = image.shape[2]
        spatial_embedding = SpatialEmbedding(bbox=bbox, document=image)
        label_encoded_list = [encoded_dic[dataloader_set.data[doc_index][1]['labels'][bb_i][1]] for bb_i in
                              range(len(dataloader_set.data[doc_index][1]['boxes']))]
        document_feat = {}
        features = []

        for bbox_index in range(len(bbox)):
            # embedding = []
            title = dataloader_set.data[doc_index][1]['labels'][bbox_index]
            # embedding.append(spatial_embedding[bbox_index])
            word_embedding, _ = word_embeder[title[0]]
            # embedding.append(word_embedding)
            image_encoder = SimpleCNNEncoder(in_channel=nbr_of_channel, out_dim=out_dim)
            xmin, ymin, xmax, ymax = bbox[bbox_index]
            data = swapaxes(swapaxes(from_numpy(image[ymin:ymax, xmin:xmax]), 0, 2), 1, 2).float()
            data = data.reshape((batch_size,) + data.shape)

            if numel(data) < out_dim:
                added_dim = out_dim - numel(data)
                encoded_data = flatten(data)
                encoded_data = pad(encoded_data, (0, added_dim), "constant", 0)
            else:
                encoded_data = flatten(image_encoder(data))
            # embedding.append(encoded_data)
            embedding_tensor = stack((spatial_embedding[bbox_index], encoded_data, word_embedding))
            features.append(embedding_tensor)
        document_feat["boxes_features"] = stack(features)
        document_feat["labels_encoded"] = tensor(label_encoded_list, dtype=int16)
        data_encoded.append(document_feat)

    return data_encoded


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def remove_duplication(array):
    index_list = []
    for i in range(len(array)):
        if array[i][0] == array[i][1]:
            index_list.append(i)
        else:
            for j in range(i, len(array)):
                if array[i][0] == array[j][1] and array[j][0] == array[i][1]:
                    index_list.append(i)
    array = np.delete(array, index_list, axis=0)
    return array


def edge_builder(data_encoded):
    # src_node = []
    # dst_node = []
    edge_array = []
    logger.debug(f"edge builded while start")
    for doc_index in range(len(data_encoded)):
        logger.debug(f"edge builded document index: {doc_index}")
        # TODO: this one take alot of resource
        nbr_bbox = data_encoded[doc_index]["boxes_features"].shape[0]
        edge_array_per_doc = remove_duplication(cartesian_product(np.arange(nbr_bbox), np.arange(nbr_bbox)))
        # src_bbox = [node_src[0] for node_src in edge_array_per_doc]
        # dst_bbox = [node_src[1] for node_src in edge_array_per_doc]
        logger.debug(f"edge builded document index after remove duplication: {doc_index}")
        # src_node.append(src_bbox)
        # dst_node.append(dst_bbox)
        edge_array.append(edge_array_per_doc)

    return edge_array


def edge_metric_calculation(edge_array, node_feature, doc_shape):
    """
    it contains:
    [
    Euclidean distance between two node (two bbox) (spatial),
    Chebyshev distance (spatial),
    Manhattan distance (spatial),
    Mutual Information (image),
    Correlation coefficient (image),
    Kullback-leibler distance (image),
    Cosine Similarity (text),
    Levenshtein distance (text),
    Point-wise Mutual information (text)
    ]
    it return: Top edge sorted by the custom metric
    :param edge_array:
    :type edge_array:
    :param node_feature:
    :type node_feature:
    :return:
    :rtype:
    """
    edge_metric_dict = {}
    logger.debug(f"len feature {len(node_feature)}")
    # for node in range(len(node_feature)):
    #   i+=1
    for edge_link in edge_array:
        logger.debug(f"the edge link {edge_link[0]} and {edge_link[1]}")
        logger.debug(f"node feature size {node_feature.shape}")
        node_src_feat = node_feature[edge_link[0] - 1]
        node_dst_feat = node_feature[edge_link[1] - 1]
        logger.debug(f"node src feat: \n {node_src_feat[0][4]} \n its related type {type(node_src_feat[0][4])}")
        # node_src_center = (node_src_feat[0][4].detach().numpy(), node_src_feat[0][5].detach().numpy())
        # node_dst_center = (node_dst_feat[0][4].detach().numpy(), node_dst_feat[0][5].detach().numpy())

        node_src_center = (
        (node_src_feat[0][0] + node_src_feat[0][7]).detach().numpy(), node_src_feat[0][1].detach().numpy())
        node_dst_center = (node_dst_feat[0][0].detach().numpy(), node_dst_feat[0][1].detach().numpy())

        node_src_image = node_src_feat[1].detach().numpy()
        node_dst_image = node_dst_feat[1].detach().numpy()

        node_src_text = node_src_feat[2].detach().numpy()
        node_dst_text = node_dst_feat[2].detach().numpy()

        # Spatial Metrics
        euclidean_dist_norm = euclidean(node_src_center, node_dst_center) / euclidean((0, 0), doc_shape)
        chebyshev_dist_norm = chebyshev(node_src_center, node_dst_center) / chebyshev((0, 0), doc_shape)
        manhattan_dist_norm = cityblock(node_src_center, node_dst_center) / cityblock((0, 0), doc_shape)

        # Visual (image) Metrics
        MI_norm = normalized_mutual_info_score(node_src_image, node_dst_image)
        correlation_coef = correlation(node_src_image, node_dst_image)

        # Textual Metrics
        cosine_similarity_norm = cosine(node_src_text, node_dst_text) / 2.0

        # TODO: this metric is very relative you can choose another strategy (may be based on weights)

        metric = ((
                              1 - euclidean_dist_norm) + chebyshev_dist_norm + manhattan_dist_norm + MI_norm + correlation_coef + cosine_similarity_norm) / 6

        edge_metric_dict[(edge_link[0], edge_link[1])] = metric
    top_dict = {}
    for node_index in range(len(node_feature)):
        node_index_dict = {}
        key_list = list(edge_metric_dict)
        for key in key_list:
            if node_index in key:
                key_list.remove(key)
                node_index_dict[key] = edge_metric_dict[key]
        # Sorting the dict by value
        sorted_dict = dict(sorted(node_index_dict.items(), key=lambda item: item[1]))
        # TODO: The max number of nodes is 4
        max_nodes = 4

        if len(sorted_dict) > max_nodes:
            for i in list(sorted_dict)[:max_nodes]:
                top_dict[i] = sorted_dict[i]
        else:
            top_dict = sorted_dict
    return top_dict
