import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset


# 假设有一个数据集类 MyDataset，你需要根据你的实际情况自定义这个类
class SystemDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return [torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])]


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, predictions, targets):
        # 计算平均绝对误差
        mae = torch.mean(torch.abs(predictions - targets))
        return mae


class SystemModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc_position = nn.Linear(50, 2)
        self.fc_radis = nn.Linear(50, 2)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc_position.weight)
        nn.init.xavier_normal_(self.fc_radis.weight)

        self.criterion = nn.HuberLoss()

    def forward(self, x):
        output1 = self.tanh(self.fc(x))
        output2 = self.tanh(self.fc2(output1))
        output3 = self.tanh(self.fc3(output2))
        position = self.fc_position(output3)
        radis = self.fc_radis(output3)
        return position, radis

    def compute_loss(self, preds, labels):
        position, radis = preds
        labels_loc = labels[:, :2]
        labels_rad = labels[:, 2:]
        loss_position = self.criterion(position, labels_loc)
        loss_rad = self.criterion(radis, labels_rad)
        loss = loss_position + loss_rad
        return loss
