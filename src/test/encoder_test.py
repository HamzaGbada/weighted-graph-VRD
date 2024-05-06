import unittest
import torch
from matplotlib import image, pyplot, patches
from numpy import asarray

from src.DataLoader.funsd_dataloader import FUNSD
from src.Encoder.image_embedding import SimpleCNNEncoder, VGG16Encoder, ResidualBlock, ResNet18Encoder
from src.Encoder.spatial_embedding import SpatialEmbedding
from src.Encoder.word_embedding import Doc2VecEncoder
from src.utils.setup_logger import logger


class TestEmbeddingEncoder(unittest.TestCase):
    def test_embedding_cnn_shape(self):
        batch_size = 256
        width = 50
        height = 58
        out_dim = 4
        # the number of channel here is the same as in the input of the first conv layer
        nbr_of_channel = 2
        images = torch.rand(batch_size, nbr_of_channel, width, height)

        image_encoder = SimpleCNNEncoder(in_channel=nbr_of_channel, out_dim=out_dim)

        encoded_data = image_encoder(images)
        logger.debug(f'data shape {encoded_data.shape}')
        self.assertEqual(encoded_data.shape, torch.Size([256, out_dim]))

    def test_embedding_vgg16_shape(self):
        batch_size = 256
        width = 48
        height = 36
        out_dim = 96
        # the number of channel here is the same as in the input of the first conv layer
        nbr_of_channel = 3
        images = torch.rand(batch_size, nbr_of_channel, width, height)
        image_encoder = VGG16Encoder(in_channel=nbr_of_channel, out_dim=out_dim)

        encoded_data = image_encoder(images)

        self.assertEqual(encoded_data.shape, torch.Size([256, out_dim]))

    def test_embedding_resnet18_shape(self):
        batch_size = 256
        width = 48
        height = 36
        out_dim = 96
        # the number of channel here is the same as in the input of the first conv layer
        nbr_of_channel = 3
        images = torch.rand(batch_size, nbr_of_channel, width, height)

        image_encoder = ResNet18Encoder(nbr_of_channel, ResidualBlock, out_dim=out_dim)

        encoded_data = image_encoder(images)

        self.assertEqual(encoded_data.shape, torch.Size([256, out_dim]))

    def test_embedding_word2vec_funsd(self):
        out_dim = 25
        x = Doc2VecEncoder("../../data/funsd_text.csv", out_dim=out_dim)
        logger.debug(x)
        word_embedding, word_similarity = x['date']
        logger.debug("word embedding of date: {}".format(word_embedding))
        logger.debug("word similarity of date: {}".format(word_similarity))

        # self.assertEqual(encoded_data.shape, torch.Size([256, out_dim]))

    def test_embedding_word2vec_cord(self):
        out_dim = 25
        word_embedding, word_similarity = Doc2VecEncoder("../../data/cord_text.csv", out_dim=out_dim)['to']
        logger.debug("word embedding of date: {}".format(word_embedding))
        logger.debug("word similarity of date: {}".format(word_similarity))

    def test_spatial_embedding(self):
        train_set = FUNSD(train=True, download=True)
        img_name = train_set.data[0][0]
        img = asarray(image.imread("/home/bobmarley/.cache/doctr/datasets/funsd/dataset/training_data/images/" + img_name))
        # fig, ax = pyplot.subplots()
        # ax.imshow(img, cmap='gray')
        logger.debug(f"image data type checker: {type(img)}")
        logger.debug(f"the first bbox of this image: {train_set.data[0][1]['boxes'][0]}")
        logger.debug(f"Type of bbox collection: {type(train_set.data[0][1]['boxes'])}")
        bbox = train_set.data[0][1]['boxes']
        spatial_embedding = SpatialEmbedding(bbox=bbox, document=img)
        logger.debug(f"spatial embedding of the first bbox Shape: {spatial_embedding.embedding[10].shape[0]}")
        logger.debug(f"spatial embedding of the first bbox: {spatial_embedding.embedding[10]}")
        # to draw a line from (200,300) to (500,100)
        # logger.debug(f"image shape: {img.shape}")
        # img_center = [img.shape[1] // 2, img.shape[0] // 2]
        # logger.debug(f"image center: {img_center}")
        # bbox_center = [spatial_embedding.embedding[10][4], spatial_embedding.embedding[10][5]]
        # xmin = spatial_embedding.embedding[10][0]
        # ymin = spatial_embedding.embedding[10][1]
        # xmax = spatial_embedding.embedding[10][2]
        # ymax = spatial_embedding.embedding[10][3]
        # w = spatial_embedding.embedding[10][6]
        # h = spatial_embedding.embedding[10][7]
        # line_point_x = [img_center[0], bbox_center[0]]
        # line_point_y = [img_center[1], bbox_center[1]]
        # rect = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor='y', facecolor='none')
        # ax.add_patch(rect)
        # pyplot.plot(img_center[0], img_center[1], marker='v', color="green")
        # pyplot.plot(bbox_center[0], bbox_center[1], marker='v', color="blue")
        # pyplot.plot(line_point_x, line_point_y, color="red", linewidth=3)
        # pyplot.show()


if __name__ == '__main__':
    unittest.main()
