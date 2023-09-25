import torch
import torch.nn as nn
import torch.nn.functional as F

class HARConvLSTM(nn.Module):
    def __init__(self, init_channels, num_classes):
        super(HARConvLSTM, self).__init__()

        self.conv1 = nn.Conv1d(init_channels, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, 5, padding=2)
        self.conv4 = nn.Conv1d(64, 64, 5, padding=2)

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        self.fc = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.transpose(x, 1, 2)
        x,(h, c) = self.lstm(x)

        h = h[-1,:,:]
        h = self.dropout(h)
        logits = self.fc(h)

        return logits


class HARBiLSTM(nn.Module):
    def __init__(self, init_channels, num_classes):
        super(HARBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=init_channels, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(128*2, num_classes)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x,(h, c) = self.lstm(x)
        h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        logits = self.fc(h)

        return logits

class HARLSTM(nn.Module):
    def __init__(self, init_channels, num_classes):
        super(HARLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=init_channels, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x,(h, c) = self.lstm(x)
        h = h[-1,:,:]
        logits = self.fc(h)

        return logits
