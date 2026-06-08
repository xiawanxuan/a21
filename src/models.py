import torch
import torch.nn as nn
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SpectrumEmbedding(nn.Module):
    def __init__(self, in_channels=1, d_model=128, kernel_size=3, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, d_model, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x


class RedshiftRegressionHead(nn.Module):
    def __init__(self, d_model, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class SpectrumTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spectrum_length = config.spectrum_length
        self.d_model = config.d_model
        self.num_classes = config.num_classes

        self.embedding = SpectrumEmbedding(
            in_channels=1, d_model=config.d_model, kernel_size=3, stride=2
        )

        seq_len = config.spectrum_length // 2
        self.pos_encoder = PositionalEncoding(config.d_model, max_len=seq_len, dropout=config.dropout)

        self.transformer_encoder = TransformerEncoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )

        self.redshift_head = RedshiftRegressionHead(config.d_model, hidden_dim=config.d_model // 2, dropout=config.dropout)
        self.classification_head = ClassificationHead(
            config.d_model, config.num_classes, hidden_dim=config.d_model // 2, dropout=config.dropout
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.embedding(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        cls_output = x[:, 0, :]

        redshift_pred = self.redshift_head(cls_output)
        class_pred = self.classification_head(cls_output)

        return redshift_pred, class_pred

    def extract_features(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x


def build_model(config):
    model = SpectrumTransformer(config)
    return model


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
