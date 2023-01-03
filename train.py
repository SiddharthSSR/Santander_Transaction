import torch
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import get_predictions, get_submission
from dataset import get_data
from torch.utils.data import DataLoader
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(NN, self).__init__()
        
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(input_size//2*hidden_dim, 1)

        # self.net = nn.Sequential(
        #     nn.BatchNorm1d(input_size),
        #     nn.Linear(input_size, 50),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(50, 1),
        # )

    def forward(self, x):
        N = x.shape[0]
        x = self.bn(x)
        org_feat = x[:, :200].unsqueeze(2) # (N, 200, 1)
        new_feat = x[:, 200:].unsqueeze(2) # (N, 200, 1)
        x = torch.cat([org_feat, new_feat], dim=2) # (N, 200, 2)
        x = F.relu(self.fc1(x)).reshape(N, -1) # (N, 200*16)
        return torch.sigmoid(self.fc2(x)).view(-1)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = NN(input_size=400, hidden_dim=128).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
loss_fn = nn.BCELoss()

train_ds, val_ds, test_ds, test_ids = get_data()

train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)

for epoch in range(20):

    probs, true = get_predictions(val_loader, model, DEVICE)
    print(f"Validation ROC: {metrics.roc_auc_score(true, probs)}")

    for batch_idx, (data, targets) in enumerate(train_loader):
    # data, targets = next(iter(train_loader))
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward
        scores = model(data)
        # print(scores.shape)
        loss = loss_fn(scores, targets)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


x, y = next(iter(train_loader))
print(x.shape)

get_submission(test_loader, model, test_ids, DEVICE)