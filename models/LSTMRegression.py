import torch.nn as nn
import torch
from control.Enums import TrainingMode4LSTMAE
from models.LSTMAE import Encoder as LSTM
from models.MLP import MLP2, MLP
from torch import Tensor


class LSTMRegression(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        lstm_layer_num=2,
        name="global;LSTMDIS",
    ) -> None:
        super(LSTMRegression, self).__init__()
        self.name = name
        self.encoder = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=0,
            training_mode=TrainingMode4LSTMAE.SPECIAL_UNCONDITIONED.value,
            encoder_only=True,
            layer_num=lstm_layer_num,
        )
        self.fc1 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight.data, mean=0, std=0.1)

    def forward(self, x):
        # print(x)
        # print(torch.transpose(x, 0, 1))
        hidden: Tensor = self.encoder(x)
        # print(hidden.shape)
        # assert False
        # hidden = hidden.squeeze(0)

        # hidden = torch.std(hidden, dim=1, keepdim=True)
        # print(hidden)
        # assert False
        out = self.fc1(hidden)
        # out = torch.sigmoid(out)
        # assert False
        return out

    def set_name(self, name):
        self.name = name
