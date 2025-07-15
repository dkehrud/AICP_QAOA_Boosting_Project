import pandas as pd
import ast
import random
import os
import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


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


def train(model, train_dataset, val_dataset, epochs, device='cpu', checkpoint_dir='132_GNN_ckpts'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    print("Training Start")
    for epoch in range(epochs):
        # 훈련 단계
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

        print(f"Epoch {epoch + 1:3d}: Train Loss {train_loss:.4f}: Val Loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'GNN_param_best_model.pth'))

    # 최종 모델 저장
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'GNN_param_final_model.pth'))
    print("Training Finished")


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
    VAL_RATIO = 0.2

    graphs, targets = load_data(DEPTH, 2, 8)
    test_graphs, test_targets = load_data(DEPTH, 9, 9)

    indices = list(range(len(graphs)))
    random.shuffle(indices)

    split_point = int(len(indices) * (1 - VAL_RATIO))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_graphs = [graphs[i] for i in train_indices]
    train_targets = [targets[i] for i in train_indices]

    val_graphs = [graphs[i] for i in val_indices]
    val_targets = [targets[i] for i in val_indices]

    print(f"Total data: {len(graphs)}, Train data: {len(train_graphs)}, Validation data: {len(val_graphs)}")

    train_dataset = GraphBetaGammaDataset(train_graphs, train_targets)
    val_dataset = GraphBetaGammaDataset(val_graphs, val_targets)

    model = GINNet(out_dim=2 * DEPTH)

    train(model, train_dataset, val_dataset, EPOCH,checkpoint_dir="132_GNN_ckpts_2to9")