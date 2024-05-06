import torch
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
from dgl.nn.pytorch import GraphConv, SAGEConv, EdgeWeightNorm
from torch import arange
from torch.nn import Module, ModuleList, Dropout
from torch.nn.functional import relu
from tqdm import tqdm


class WGCN(Module):
    def __init__(self, n_infeat, n_hidden, n_classes, n_layers, activation):
        super(WGCN, self).__init__()
        self.layers = ModuleList()
        self.layers.append(GraphConv(n_infeat, n_hidden, weight=True, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, weight=True, activation=activation))
        self.layers.append(GraphConv(n_hidden, n_classes, weight=True,))

        self.edge_norm = EdgeWeightNorm(norm='both')

    def forward(self, g, features, edge_weight):
        h = features
        norm_edge_weight = self.edge_norm(g, edge_weight)
        for i, layer in enumerate(self.layers):
            # handle api changes for differnt DGL version
            h = layer(g, h, edge_weight = norm_edge_weight)
        return h


class StochasticTwoLayerGCN(Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = GraphConv(in_features, hidden_features)
        self.conv2 = GraphConv(hidden_features, out_features)

    def forward(self, blocks, x):
        x = relu(self.conv1(blocks[0], x))
        x = relu(self.conv2(blocks[1], x))
        return x

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y

# TODO: setup a node classifier with all these method:
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.TAGConv.html
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.EGATConv.html
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.SAGEConv.html
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.SGConv.html
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.APPNPConv.html
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.GINConv.html
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.GINEConv.html
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.CFConv.html
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.GCN2Conv.html
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.EGNNConv.html
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.PNAConv.html
#         - https://docs.dgl.ai/en/0.9.x/generated/dgl.nn.pytorch.conv.DGNConv.html

class SAGE(Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = ModuleList()
        self.layer1 = SAGEConv(in_size, hid_size, 'mean')
        self.layer2 = SAGEConv(hid_size, hid_size, 'mean')
        self.layer3 = SAGEConv(hid_size, out_size, 'mean')
        self.dropout = Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = relu(h)
        h = self.dropout(h)

        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
            g, arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y
