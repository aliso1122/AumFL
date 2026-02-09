import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # 双向LSTM捕捉前后依赖
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 自注意力机制捕捉重要时间点
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim*2,  # 双向
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 时序卷积提取局部模式
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(hidden_dim*2, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 对于原始GAN
        )
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        # LSTM提取时序特征
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # 自注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 转置用于卷积
        conv_input = attn_out.transpose(1, 2)  # (batch, hidden*2, seq_len)
        
        # 卷积提取局部特征
        conv_out = self.conv_blocks(conv_input)
        
        # 全局池化
        pooled = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)
        
        # 分类
        output = self.classifier(pooled)
        return output