import pandas as pd
import ast
import os
import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

class GraphBetaGammaDataset(InMemoryDataset):
    def __init__(self, graphs, targets, transform=None):
        self.graphs, self.targets = graphs, targets
        super().__init__('.', transform)
        items = [self._to_data(g, y) for g, y in zip(graphs, targets)]
        self.data, self.slices = self.collate(items)

    def _to_data(self, G, y):
        n, edge_list = G
        x = torch.ones((n, 1), dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
        y = torch.tensor([y], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)

class GINNet(nn.Module):
    def __init__(self, hidden=128, num_layers=10, out_dim=6):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden if i else 1, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            outputs.append(x)
        x = sum(outputs)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train(model, train_dataset, val_dataset, epochs, device='cpu', checkpoint_dir=None, fold_number=None, silent=False):
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optim.step()
            train_loss = loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_loss = criterion(pred, batch.y).item()

        if not silent:
            print(f"Epoch {epoch + 1:3d}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_dir:
                filename = f'best_model_fold_{fold_number}.pth' if fold_number is not None else 'best_model_overall.pth'
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, filename))

    if checkpoint_dir:
        filename = f'final_model_fold_{fold_number}.pth' if fold_number is not None else 'final_model_overall.pth'
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, filename))

    if not silent:
        print(f"Best validation loss for this run: {best_val_loss:.4f}")

    return best_val_loss


def load_data(depth, start_node_count, end_node_count):
    dataset_x = []
    dataset_y = []

    for i in range(start_node_count, end_node_count + 1):
        data_x = []
        graph_data = pd.read_csv(f"GNN_data_csv/graphs_csv/graph{i}c.csv")
        graph_result_data = pd.read_csv(f"GNN_data_csv/result_p_{depth}_csv/n={i}_p={depth}.csv")
        graph_connection_data = graph_data['connection_data'].values.tolist()
        graph_node_data = graph_data['graph_node'].values.tolist()
        for idx, element in enumerate(graph_connection_data):
            data_x.append([graph_node_data[0], ast.literal_eval(element)])

        data_y = graph_result_data.iloc[:, -(2 * depth):].to_numpy().tolist()
        dataset_x += data_x
        dataset_y += data_y

    return dataset_x, dataset_y


if __name__ == "__main__":
    EPOCH = 50
    DEPTH = 1
    N_SPLITS = 10
    CHECKPOINT_DIR = "132_GNN_10f_model_ckpts"

    all_train_graphs, all_train_targets = load_data(DEPTH, 2, 8)
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_validation_losses = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_train_graphs)):
        fold_num = fold + 1
        print(f"fold {fold_num}/{N_SPLITS}")

        train_graphs = [all_train_graphs[i] for i in train_ids]
        train_targets = [all_train_targets[i] for i in train_ids]
        val_graphs = [all_train_graphs[i] for i in val_ids]
        val_targets = [all_train_targets[i] for i in val_ids]

        train_dataset = GraphBetaGammaDataset(train_graphs, train_targets)
        val_dataset = GraphBetaGammaDataset(val_graphs, val_targets)

        model = GINNet(out_dim=2 * DEPTH)

        best_loss = train(model, train_dataset, val_dataset, EPOCH, checkpoint_dir=CHECKPOINT_DIR, fold_number=fold_num, silent=True)
        fold_validation_losses.append(best_loss)
        print(f"Fold {fold_num} bast val loss: {best_loss:.4f}")

    avg_loss = np.mean(fold_validation_losses)
    std_loss = np.std(fold_validation_losses)

    print(f"val loss: {[f'{loss:.4f}' for loss in fold_validation_losses]}")
    print(f"avg vall loss: {avg_loss:.4f}")
    print(f"val loss stdev: {std_loss:.4f}")

    full_train_dataset = GraphBetaGammaDataset(all_train_graphs, all_train_targets)
    test_graphs, test_targets = load_data(DEPTH, 9, 9)
    test_dataset = GraphBetaGammaDataset(test_graphs, test_targets)

    final_model = GINNet(out_dim=2 * DEPTH)

    #final training
    train(final_model, full_train_dataset, test_dataset, EPOCH, checkpoint_dir=CHECKPOINT_DIR, fold_number=None, silent=False)