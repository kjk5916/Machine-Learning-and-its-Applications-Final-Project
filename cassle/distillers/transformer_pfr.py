import argparse
from typing import Any, List, Sequence

import torch
from torch import nn
from cassle.distillers.base import base_distill_wrapper

from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch.autograd import detect_anomaly

def transformer_pfr_distill_wrapper(Method=object):
    class TransformerPfrDistillWrapper(base_distill_wrapper(Method)):
        def __init__(
            self,
            distill_lamb: float,
            distill_proj_hidden_dim: int,
            distill_barlow_lamb: float,
            distill_scale_loss: float,
            **kwargs
        ):
            super().__init__(**kwargs)

            output_dim = kwargs["output_dim"]
            self.distill_lamb = distill_lamb
            self.distill_barlow_lamb = distill_barlow_lamb
            self.distill_scale_loss = distill_scale_loss

            encoder_layer = TransformerEncoderLayer(
                d_model = 26,
                nhead= 2,
                dim_feedforward = 26 * 4,
                dropout=0.1,
                batch_first=True,
                activation = 'gelu'
            )
            
            self.distill_predictor = TransformerEncoder(
                encoder_layer,
                num_layers=1
            )

            # self.trans_projector = nn.Linear(25, 255)

            self.cls_token = nn.parameter.Parameter(torch.rand(1, 512, 1))

            # Transformer_PFR criterion
            self.criterion = nn.MSELoss()
            
            # PFR criterion
            # self.criterion = nn.CosineSimilarity(dim=1)
        
        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("contrastive_distiller")

            parser.add_argument("--distill_lamb", type=float, default=25)
            parser.add_argument("--distill_proj_hidden_dim", type=int, default=2048)
            parser.add_argument("--distill_barlow_lamb", type=float, default=5e-3)
            parser.add_argument("--distill_scale_loss", type=float, default=0.1)

            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            """Adds distill predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {"params": self.distill_predictor.parameters()},
                {"params": self.cls_token},
                # {"params": self.trans_projector.parameters()},
            ]
            return super().learnable_params + extra_learnable_params

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
                        
            out = super().training_step(batch, batch_idx)
            
            f1, f2 = out["feats"]
            # print("f1.shape:", f1.shape)
            # f1, f2 has size batch_size x 512 x 5 x 5
            # print(f"encoder output size: {f1.shape}")
            frozen_f1, frozen_f2 = out["frozen_feats"]

            f1 = f1.view(f1.size(0), 512, -1)
            f2 = f2.view(f2.size(0), 512, -1)
            frozen_f1 = frozen_f1.view(frozen_f1.size(0), 512, -1)
            frozen_f2 = frozen_f2.view(frozen_f2.size(0), 512, -1)
            # All the output sizes are batch_size x 512 x 25

            # f1 = self.trans_projector(f1)
            # f2 = self.trans_projector(f2)
            
            # print(f'f1 shape {f1.shape}')
            # f1 has size batch_size x 512 x 255

            cls_token_expanded = self.cls_token.expand(f1.size(0), -1, -1)
            f1 = torch.cat((cls_token_expanded, f1), dim=2)
            f2 = torch.cat((cls_token_expanded, f2), dim=2)

            # print(f'f1 shape: {f1.shape}')
            # f1 has size batch_size x 512 x 26

            p1 = self.distill_predictor(f1)[:,:,0]
            p2 = self.distill_predictor(f2)[:,:,0]
            # self.distill_predictor(f1) has batch_size x 512 x 26 with the added cls token
            
            # print(f'p1 shape: {p1.shape}')
            # p1 has size batch_size x 512

            # p1 = torch.mean(p1, dim=2)
            # p2 = torch.mean(p2, dim=2)
            
            frozen_f1 = torch.mean(frozen_f1, dim=2)
            frozen_f2 = torch.mean(frozen_f2, dim=2)
            # All the output sizes are batch_size x 512

            # Transformer PFR Distill Loss
            distill_loss = self.criterion(p1, frozen_f1.detach()) + self.criterion(p2, frozen_f2.detach()) / 2

            # PFR Distill Loss
            # distill_loss = -1*(self.criterion(p1, frozen_f1.detach()).mean() + self.criterion(p2, frozen_f2.detach()).mean() * 0.5)


            self.log(
                "train_decorrelative_distill_loss", distill_loss, on_epoch=True, sync_dist=True
            )

            return out["loss"] + self.distill_lamb * distill_loss

    return TransformerPfrDistillWrapper
