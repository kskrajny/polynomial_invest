import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, random_split
import torch.optim as optim


def map_(x: bool):
    return 1 if x else -1


def predict_with_treshold(p, t):
    if p <= 1 - t:
        return 0
    elif p >= t:
        return 1
    else:
        return -2


def statistics_universal(self, X, y, potential_gains, sample_weight=None, treshold=0.5):
    proba = self.predict_proba(X)
    pred = np.array([predict_with_treshold(p[1], treshold) for p in proba], dtype=np.float)
    measurable_y = y[np.all([y != -1, pred != -2], axis=0)].flatten()
    measurable_pred = pred[np.all([y != -1, pred != -2], axis=0)].flatten()
    lost_test = len(y) - len(measurable_y)
    gains = [
        int(map_(t == p) * g * 10000) if p != -2 else 0 for t, p, g in zip(y, pred, potential_gains)
    ]
    return (
        accuracy_score(measurable_y, measurable_pred, sample_weight=sample_weight),
        confusion_matrix(measurable_y, measurable_pred, labels=None),
        lost_test,
        gains
    )


class CustomDataset(Dataset):
    def __init__(self, X, y=None):
        super().__init__()
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return torch.tensor(self.X[idx])
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class ANN(nn.Module):
    def __init__(self, dims, device='cuda:0'):
        super().__init__()
        self.batch_size = 256
        self.max_epochs = 200
        self.device = device
        self.loss_function = nn.CrossEntropyLoss()
        self.activation = nn.ReLU()
        self.net = nn.ModuleList()
        for i in range(len(dims) - 2):
            self.net.append(nn.Linear(dims[i], dims[i+1]))
            self.net.append(self.activation)
        self.net.append(nn.Linear(dims[-2], dims[-1]))
        self.net.append(nn.Softmax(dim=-1))
        self.net.to(device)

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x

    def fit(self, X_train, y_train):
        last_loss = 100
        patience = 3
        trigger_times = 0
        optimizer = optim.Adam(self.parameters(), lr=.0001)
        dataset = CustomDataset(X_train, y_train)
        train_size = int(len(dataset) * 0.8)
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        for epoch in range(1, self.max_epochs):
            self.train()
            for times, data in enumerate(train_loader):
                x, y = data
                optimizer.zero_grad()
                pred = self(x.to(self.device).float())
                loss = self.loss_function(pred, y.to(self.device).long())
                loss.backward()
                optimizer.step()
            current_loss = self._validation(valid_loader)
            if current_loss > last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    return self
            else:
                trigger_times = 0
            last_loss = current_loss
        return self

    def statistics(self, X, y, potential_gains, sample_weight=None, treshold=0.5):
        return statistics_universal(self, X, y, potential_gains, sample_weight=sample_weight, treshold=treshold)

    def _validation(self, valid_loader):
        self.eval()
        loss_total = 0
        with torch.no_grad():
            for data in valid_loader:
                x, y = data
                pred = self(x.to(self.device).float())
                loss = self.loss_function(pred, y.to(self.device).long())
                loss_total += loss.item()
        return loss_total / len(valid_loader)

    def predict_proba(self, X):
        self.eval()
        dataset = CustomDataset(X)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for data in data_loader:
                preds.append(self(data.to(self.device).float()).cpu().numpy())
        return np.concatenate(preds)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class CustomTabNet(TabNetClassifier):
    def __init__(self):
        super().__init__(verbose=0, n_d=64, n_a=64)
        self.softmax = nn.Softmax(dim=-1)

    def statistics(self, X, y, potential_gains, sample_weight=None, treshold=0.5):
        return statistics_universal(self, X, y, potential_gains, sample_weight=sample_weight, treshold=treshold)

    def fit(self, X_train, y_train, *kwargs):
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, test_size=0.2,
                                                              shuffle=True)

        unsupervised_model = TabNetPretrainer(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=.0001),
            mask_type='entmax',
            verbose=0,
            n_d=64,
            n_a=64
        )
        unsupervised_model.fit(
            X_train=X_train,
            eval_set=[X_valid],
            batch_size=256,
            max_epochs=200,
            pretraining_ratio=0.8
        )
        super(TabNetClassifier, self).fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=['auc'],
            batch_size=256,
            max_epochs=200,
            from_unsupervised=unsupervised_model,
        )
        return self

    def predict_proba(self, X):
        self.network.eval()
        dataset = CustomDataset(X)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for data in data_loader:
                preds.append(self.softmax(self.network(data.to(self.device).float())[0]).cpu().numpy())
        return np.concatenate(preds)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
