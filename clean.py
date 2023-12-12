import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import heads
from lightly.utils.benchmarking import BenchmarkModule
from lightly.utils.scheduler import CosineWarmupScheduler

from lamb import Lamb
from simple_vit_1d import SimpleViT


class DualSimpleViT(nn.Module):
    def __init__(self, dim, packets, subcarries, depth, heads, mlp_dim, dim_head=64):
        super().__init__()
        self.backbone_one = SimpleViT(
            packets,
            subcarries,
            depth,
            heads,
            mlp_dim,
            dim_head,
        )
        self.backbone_two = SimpleViT(
            packets,
            subcarries,
            depth,
            heads,
            mlp_dim,
            dim_head,
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(2 * subcarries),
            nn.Linear(2 * subcarries, 4 * subcarries),
            nn.GELU(),
            nn.Linear(4 * subcarries, dim),
        )

    def forward(self, x):
        x_one = self.backbone_one(x[:, 0]).flatten(start_dim=1)
        x_two = self.backbone_two(x[:, 1]).flatten(start_dim=1)
        x = torch.cat([x_one, x_two], dim=1)
        return self.to_out(x)


class CLEAN(BenchmarkModule):
    def __init__(
        self,
        lr_factor,
        packets,
        subcarries,
        dataloader_kNN,
        num_classes,
        knn_k: int,
        knn_t: float = 0.1,
    ):
        super().__init__(dataloader_kNN, num_classes, knn_k, knn_t)
        self.lr_factor = lr_factor
        self.backbone = DualSimpleViT(
            64,
            packets,
            subcarries,
            depth=8,
            heads=4,
            mlp_dim=256,
            dim_head=16,
        )
        self.projection_head = heads.ProjectionHead(
            [
                (64, 256, nn.BatchNorm1d(256), nn.ReLU(inplace=True)),
                (256, 256, nn.BatchNorm1d(256), nn.ReLU(inplace=True)),
                (256, 256, nn.BatchNorm1d(256, affine=False), None),
            ]
        )

        self.criterion = BarlowTwinsLoss(gather_distributed=False)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        views = batch
        features = torch.stack([self.forward(view) for view in views])

        l2_norm = F.normalize(features, p=2, dim=-1)
        reduce_std = torch.mean(torch.std(l2_norm, dim=1))
        self.log("mean", reduce_std, prog_bar=True)

        loss = 0.0
        for i in range(len(views)):
            mask = torch.arange(len(views), device=self.device) != i
            loss += self.criterion(
                features[i],
                torch.mean(features[mask], dim=0),
            ) / len(views)

        self.log("train_loss_ssl", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        optim = Lamb(
            self.parameters(),
            lr=1e-4 * self.lr_factor,
            weight_decay=1e-3,
            trust_clip=True,
            always_adapt=True,
        )
        scheduler = CosineWarmupScheduler(optim, self.warmup_epochs, self.max_epochs)
        return [optim], [scheduler]
