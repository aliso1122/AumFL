import torch.nn as nn
import torch
from typing import List
from control.Enums import TrainingMode4LSTMAE

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AEwithAux(nn.Module):
    def __init__(
        self,
        encoder_list: List[nn.Module],
        decoder_list: List[nn.Module],
        aux_models: List[nn.Module],
        name="glboal;ae-aux",
    ) -> None:
        super(AEwithAux, self).__init__()
        self.encoder_list = encoder_list
        self.decoder_list = decoder_list
        self.aux_models = aux_models
        self.name = name

    def forward(
        self,
        x: torch.Tensor,
        encoder_id: int,
        decoder_id: int,
        aux_id: int,
        to_transpose: int = 0,
    ):
        # self.extractor.train()
        encoder = self.encoder_list[encoder_id]
        decoder = self.decoder_list[decoder_id]

        aux_model = self.aux_models[aux_id]
        rep = encoder(x)

        des_x = decoder(rep)

        # if to_transpose == 1:
        #     des_x = torch.transpose(des_x, 0, 1)

        # print(des_x)
        out = aux_model(des_x)

        return out
