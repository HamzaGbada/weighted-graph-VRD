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
        logits = model(g, features, edge_weight)

        pred = logits.argmax(1)

        loss = cross_entropy(logits[train_mask], labels[train_mask])
        loss_train.append(loss.to('cpu').detach().numpy())
        loss_val.append(cross_entropy(logits[val_mask], labels[val_mask]).to('cpu').detach().numpy())
        loss_test.append(cross_entropy(logits[test_mask], labels[test_mask]).to('cpu').detach().numpy())

        train_f1 = f1_score(pred[train_mask], labels[train_mask], mdmc_average='global', num_classes=num_class,
                            multiclass=True)
        val_f1 = f1_score(pred[val_mask], labels[val_mask], mdmc_average='global', num_classes=num_class,
                          multiclass=True)
        test_f1 = f1_score(pred[test_mask], labels[test_mask], mdmc_average='global', num_classes=num_class,
                           multiclass=True)
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        train_list.append(train_f1.to('cpu'))
        val_list.append(val_f1.to('cpu'))
        test_list.append(test_f1.to('cpu'))
        if best_val_acc < val_acc:
            best_val_acc = val_acc
        if best_val_f1 < val_f1:
            best_val_f1 = val_f1

        if best_test_f1 < test_f1:
            best_test_f1 = test_f1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.debug(f"Best Test f1-score {best_test_f1}")
        if e % 10 == 0:
            logger.debug(f"Epochs: {e}/{epochs}, Train F1-score: {train_f1}, Val F1-score: {val_f1}, Train Accuracy: "
                         f"{train_acc}, Val Accuracy: {val_acc}, Best Accuracy: {best_val_acc}, Best F1-score: {best_val_f1}, Best Test F1-score: {best_test_f1}")
    return train_list, val_list, test_list, loss_train, loss_val, loss_test


if __name__ == '__main__':

    data_name = argument1.dataname
    path = argument1.path
    hidden_size = argument1.hidden_size
    nbr_hidden_layer = argument1.hidden_layers
    lr = argument1.learning_rate
    epochs = argument1.epochs

    dataset = DocumentGraphDataset(data_name, path=path)
    graph_train = dataset[True].to('cuda')
    graph_train = add_self_loop(graph_train)
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    model = WGCN(graph_train.ndata['feat'].shape[2], hidden_size, dataset.num_classes, nbr_hidden_layer, relu).to(
        'cuda')
    model.double()
    edge_weight = graph_train.edata['weight'].to('cuda')

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

