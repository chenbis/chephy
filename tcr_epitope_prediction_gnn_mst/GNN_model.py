import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
import torchmetrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HeteroTCR(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=1024, num_layers=3, net_type='SAGE'):
        super().__init__()
        self.encoder = HeteroGNN(metadata, hidden_channels, num_layers, net_type)
        self.decoder = MLP(hidden_channels)
    
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=1024, num_layers=3, net_type='SAGE'):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        if net_type == 'SAGE':
            for _ in range(num_layers):
                conv = HeteroConv({
                    edge_type: SAGEConv((-1, -1), hidden_channels)
                    for edge_type in metadata[1]
                })
                self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return x_dict


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels=1024):
        super().__init__()

        self.lin1 = Linear(hidden_channels * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.lin3 = Linear(256, 1)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x_dict, edge_label_index):
        row, col = edge_label_index
        try:
            x = torch.cat([x_dict['tcr'][row], x_dict['epitope'][col]], dim=-1)
        except:
            x = torch.cat([x_dict['tcr'][row], x_dict['peptide'][col]], dim=-1)

        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        x = self.lin3(x)
        x = self.sigmoid(x)
        return x.view(-1)


def train(model, optimizer, data_hetero_train, data_hetero_test, train_edge_label_index, test_edge_label_index, y_train, y_test, device):
    loss_fn = torch.nn.BCELoss()

    model.train()
    optimizer.zero_grad()
    out = model(data_hetero_train.x_dict, data_hetero_train.edge_index_dict, train_edge_label_index)
    train_loss = loss_fn(out, torch.tensor(y_train).float().to(device))
    train_loss.backward()
    optimizer.step()

    train_binary_accuracy = torchmetrics.functional.accuracy(out, torch.tensor(y_train).int().to(device))
    train_ROCAUC = torchmetrics.functional.auroc(out, torch.tensor(y_train).int().to(device))

    model.eval()
    with torch.no_grad():
        out_test = model(data_hetero_test.x_dict, data_hetero_test.edge_index_dict, test_edge_label_index)
        test_loss = loss_fn(out_test, torch.tensor(y_test).float().to(device))
        test_binary_accuracy = torchmetrics.functional.accuracy(out_test, torch.tensor(y_test).int().to(device))
        test_ROCAUC = torchmetrics.functional.auroc(out_test, torch.tensor(y_test).int().to(device))

    return train_loss, train_binary_accuracy, train_ROCAUC, test_loss, test_binary_accuracy, test_ROCAUC

def predict(new_model, data, edge_label_index, y):
    loss_fn = torch.nn.BCELoss()
    new_model.eval()
    with torch.no_grad():
        out_test = new_model(data.x_dict, data.edge_index_dict, edge_label_index)
        # print(out_test)
        test_loss = loss_fn(out_test, torch.tensor(y).float().to(device))
        test_binary_accuracy = torchmetrics.functional.accuracy(out_test, torch.tensor(y).int().to(device))
        test_ROCAUC = torchmetrics.functional.auroc(out_test, torch.tensor(y).int().to(device))
    return test_loss, test_binary_accuracy, test_ROCAUC, out_test