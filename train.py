import torch
from dgl import add_self_loop
from matplotlib.pyplot import plot, show, savefig, figure
from numpy import arange
from sklearn.preprocessing import LabelBinarizer
from torch.nn.functional import cross_entropy, relu
from torch.optim import Adam
from torchmetrics.functional import f1_score

from args import argument1
from src.DataLoader.graph_dataloader import DocumentGraphDataset
from src.GraphModule.node_classifier import WGCN
from src.utils.setup_logger import logger


def train(g, model, edge_weight, train_mask, val_mask, test_mask, num_class, lr=0.01, epochs=50):
    optimizer = Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_val_f1 = 0
    best_test_f1 = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(max(labels) + 1))
    labels = torch.from_numpy(label_binarizer.transform(labels.to('cpu'))).to('cuda')

    train_mask = train_mask
    val_mask = val_mask
    test_mask = test_mask
    train_list, val_list, test_list = [], [], []
    loss_train, loss_val, loss_test = [], [], []
    for e in range(epochs):
        # Forward
        logits = model(g, features, edge_weight)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        logger.debug(f"feature shape labels[train_mask]{labels[train_mask].shape}")
        logger.debug(f"feature shape logits[train_mask]{logits[train_mask].shape}")
        # TODO: the error of shape (check the output of the below) is due to the multiclass classification (change the label)
        loss = cross_entropy(logits[train_mask], labels[train_mask])
        loss_train.append(loss.to('cpu').detach().numpy())
        loss_val.append(cross_entropy(logits[val_mask], labels[val_mask]).to('cpu').detach().numpy())
        loss_test.append(cross_entropy(logits[test_mask], labels[test_mask]).to('cpu').detach().numpy())

        # Compute accuracy on training/validation/test
        train_f1 = f1_score(pred[train_mask], labels[train_mask], mdmc_average='global', num_classes=num_class,
                            multiclass=True)
        val_f1 = f1_score(pred[val_mask], labels[val_mask], mdmc_average='global', num_classes=num_class,
                          multiclass=True)
        test_f1 = f1_score(pred[test_mask], labels[test_mask], mdmc_average='global', num_classes=num_class,
                           multiclass=True)
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        train_list.append(train_f1.to('cpu'))
        val_list.append(val_f1.to('cpu'))
        test_list.append(test_f1.to('cpu'))
        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            # best_test_acc = test_acc
        if best_val_f1 < val_f1:
            best_val_f1 = val_f1

        if best_test_f1 < test_f1:
            best_test_f1 = test_f1

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.debug(f"Best Test f1-score {best_test_f1}")
        if e % 10 == 0:
            logger.debug(f"Epochs: {e}/{epochs}, Train F1-score: {train_f1}, Val F1-score: {val_f1}, Train Accuracy: "
                         f"{train_acc}, Val Accuracy: {val_acc}, Best Accuracy: {best_val_acc}, Best F1-score: {best_val_f1}, Best Test F1-score: {best_test_f1}")
    return train_list, val_list, test_list, loss_train, loss_val, loss_test


if __name__ == '__main__':
    # TODO: Add path to dataset as argument
    #       Check how to create two file of agrs in python

    data_name = argument1.dataname
    path = argument1.path
    hidden_size = argument1.hidden_size
    nbr_hidden_layer = argument1.hidden_layers
    lr = argument1.learning_rate
    epochs = argument1.epochs

    dataset = DocumentGraphDataset(data_name, path=path)
    # TODO: Apply the required tests for this section
    graph_train = dataset[True].to('cuda')
    graph_train = add_self_loop(graph_train)
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    # test_index = torch.arange(graph_test.number_of_nodes())
    logger.debug(f"data training indexes {train_mask}")
    logger.debug(f"data validiation indexes {val_mask}")
    # logger.debug(f"data testing indexes {test_index}")

    model = WGCN(graph_train.ndata['feat'].shape[2], hidden_size, dataset.num_classes, nbr_hidden_layer, relu).to(
        'cuda')
    model.double()
    edge_weight = graph_train.edata['weight'].to('cuda')
    logger.debug(f"edge weight {edge_weight}")
    logger.debug(f"edge weight shape {edge_weight.shape}")
    logger.debug(f"number of edges {graph_train.number_of_edges()}")
    logger.debug(f"number of nodes {graph_train.number_of_nodes()}")
    logger.debug(f"feature shape {graph_train.ndata['feat'].shape}")

    train_list, val_list, test_list, loss, loss_val, loss_test = train(graph_train, model, edge_weight, train_mask,
                                                                       val_mask, test_mask, dataset.num_classes, lr,
                                                                       epochs)
    x = arange(epochs)
    fig1 = figure()
    plot(x, train_list, color='b')
    plot(x, val_list, color='g')
    plot(x, test_list, color='r')
    fig1.savefig("data/" + data_name + "_f1-score.png")
    fig2 = figure()
    plot(x, loss, color='b')
    plot(x, loss_val, color='g')
    plot(x, loss_test, color='r')
    fig2.savefig("data/" + data_name + "_loss.png")
    # dataset = DocumentGraphDataset("FUNSD")
    # # TODO: Apply the required tests for this section
    # graph_train = dataset[True]
    # graph_test = dataset[False]
    # graph_train = add_self_loop(graph_train)
    # graph_train = add_self_loop(graph_train)
    # train_index = dataset.train_index
    # val_index = dataset.val_index
    # test_index = torch.arange(graph_test.number_of_nodes())
    # logger.debug(f"data training indexes {train_index}")
    # logger.debug(f"data validiation indexes {val_index}")
    # logger.debug(f"data testing indexes {test_index}")
    # mode = 'puregpu'
    # if not torch.cuda.is_available():
    #     mode = 'cpu'
    # logger.debug(f'Training in {mode} mode.')
    # epochs = 3
    # # load and preprocess dataset
    # logger.debug('Loading data')
    # # dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    #
    # graph_train = graph_train.to('cuda' if mode == 'puregpu' else 'cpu')
    # graph_test = graph_test.to('cuda' if mode == 'puregpu' else 'cpu')
    # device = torch.device('cpu' if mode == 'cpu' else 'cuda')
    #
    # # create GraphSAGE model
    # in_size = graph_train.ndata['feat'].shape[2]
    # out_size = dataset.num_classes
    # model = StochasticTwoLayerGCN(in_size, 256, out_size).to(device)
    #
    # # model training
    # logger.debug('Training...')
    # train(mode, device, graph_train, train_index, val_index, model, epochs)
    #
    # # test the model
    # logger.debug('Testing...')
    # acc = layerwise_infer(device, graph_test, test_index, model, batch_size=4096)
    # logger.debug("Test Accuracy {:.4f}".format(acc.item()))
