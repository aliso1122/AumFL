import torch.nn as nn
import torch
from control.Enums import TrainingMode4LSTMAE
from models.LSTMAE import Encoder as LSTMEncoder
from models.MLP import MLP2, MLP
from torch import Tensor
import torch.nn.functional as F

class LSTMDIS(nn.Module):
    def __init__(self, input_size, hidden_size, name="global;LSTMDIS") -> None:
        super(LSTMDIS, self).__init__()
        hidden_size = 256
        self.name = name
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=0,
            training_mode=TrainingMode4LSTMAE.SPECIAL_UNCONDITIONED.value,
            encoder_only=True,
            layer_num=2,
            encoder_only_keep_dim=False
        )
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 2)
        self.activation = nn.LeakyReLU(0.01)  # 或者 nn.ReLU()
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu', a=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        # nn.init.normal_(self.fc1.weight.data, mean=0, std=0.1)
        # nn.init.normal_(self.fc2.weight.data, mean=0, std=0.1)
        # self.activation = nn.ELU(0.2)

    def forward(self, x):

        hidden: Tensor = self.encoder(x)
        out = self.fc1(hidden)
        out = self.activation(out)
        out = self.fc2(out)
        return out

    def set_name(self, name):
        self.name = name


class TSDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_dim=64, name="global;LSTMDIS"):
        """
        轻量级时间序列判别器
        Args:
            input_dim: 输入特征维度 (对应 hidden_dim in your data)
            hidden_dim: GRU隐藏层维度，默认64保持轻量
        """
        super().__init__()
        
        # 单层双向GRU，轻量且能捕捉时序依赖
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False,  # 输入格式: (seq_len, batch, input_dim)
            bidirectional=False
        )
        
        # 输出层映射到2维
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        """
        Args:
            x: (seq_len, batch, input_dim)
        Returns:
            output: (batch, 2) - 2维向量 [score_1, score_2]
        """
        # GRU输出: output (seq_len, batch, hidden_dim), hidden (1, batch, hidden_dim)
        _, hidden = self.gru(x)
        
        # 取最后时刻的隐状态 (batch, hidden_dim)
        features = hidden.squeeze(0)
        
        # 映射到2维输出 (batch, 2)
        logits = self.fc(features)
        
        return logits
    
    def set_name(self, name):
        self.name = name