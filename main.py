import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
from einops import pack, rearrange
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

import preprocess
from clean import CLEAN
from mapping import DeviceMapping


class CSIDataset(Dataset):
    def __init__(self, dss: list, n_packets: int, ratio: float):
        super().__init__()

        self.dss = dss
        self.n_packets = n_packets
        self.n_device = len(dss[0]["sc0"]["pha"])
        self.csi_per_device = int(len(dss[0]["sc0"]["amp"][0]) * ratio)

    def __getitem__(self, index):
        views = list()

        offsets = np.random.randint(
            self.csi_per_device - self.n_packets,
            size=(len(self.dss),),
        )

        for i, ds in enumerate(self.dss):
            views.append(self.transform(ds, index, offsets[i]))

        return views

    def __len__(self):
        return self.n_device * self.csi_per_device

    def transform(self, ds: dict, index: int, offset: int):
        ds_t = list()

        Q = index // self.csi_per_device
        R = index % (self.csi_per_device - self.n_packets)
        R = (R + offset) % (self.csi_per_device - self.n_packets)

        def read_data_from_matrix(mat: np.ndarray, start_index, n_packets):
            result = mat[start_index : start_index + n_packets]

            return result

        for key in ds.keys():
            pha = read_data_from_matrix(
                ds[key]["pha"][Q][: self.csi_per_device],
                R,
                self.n_packets,
            )
            amp = read_data_from_matrix(
                ds[key]["amp"][Q][: self.csi_per_device],
                R,
                self.n_packets,
            )
            ds_t.append(rearrange([pha, amp], "i j k-> i (j k)"))

        return rearrange(ds_t, "i j k -> j k i")


class KNNDataset(Dataset):
    def __init__(self, x: dict, y: list, n_packets: int, ratio: float):
        super().__init__()
        self.ds = x
        self.labels = y
        self.n_packets = n_packets
        self.n_device = len(x["sc0"]["pha"])
        self.csi_per_device = int(len(x["sc0"]["amp"][0]) * ratio)

    def __getitem__(self, index):
        ds, label = self.transform(self.ds, self.labels, index)
        # benchmark training/validation_epoch_end unpack
        padding = 0

        return ds, label, padding

    def __len__(self):
        return self.n_device * self.csi_per_device

    def transform(self, ds: dict, labels: list, index: int):
        ds_t = list()

        Q = index // self.csi_per_device
        R = index % (self.csi_per_device - self.n_packets)

        def read_data_from_matrix(mat: np.ndarray, start_index, n_packets):
            result = mat[start_index : start_index + n_packets]

            return result

        for key in ds.keys():
            pha = read_data_from_matrix(
                ds[key]["pha"][Q][-self.csi_per_device :],
                R,
                self.n_packets,
            )
            amp = read_data_from_matrix(
                ds[key]["amp"][Q][-self.csi_per_device :],
                R,
                self.n_packets,
            )
            ds_t.append(rearrange([pha, amp], "i j k-> i (j k)"))

        label_t = labels[Q][0]

        return rearrange(ds_t, "i j k -> j k i"), label_t

    def get_num_classes(self):
        return len(self.labels)


if __name__ == "__main__":
    n_packets = 20
    n_epochs = 100
    ssl_batch_size = 1024
    knn_batch_size = 4096
    ratio = 0.8

    pl.seed_everything(41)

    com6_ds, com6_labels = preprocess.load_labeled_data(
        pkl_src="dataset/esp-com-6-pkl",
        pattern="ds-67",
        sample_per_device=None,
    )

    com7_ds, com7_labels = preprocess.load_labeled_data(
        pkl_src="dataset/esp-com-7-pkl",
        pattern="ds-67",
        sample_per_device=None,
    )

    ssl_dataset = CSIDataset([com6_ds, com7_ds], n_packets, ratio)
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=ssl_batch_size,
        shuffle=True,
        num_workers=12,
    )

    max_steps = n_epochs * len(ssl_loader)
    warmup_steps = 10 * len(ssl_loader)

    knn_train_dataset = KNNDataset(com7_ds, com7_labels, n_packets, 1 - ratio)
    knn_train_loader = DataLoader(
        knn_train_dataset,
        batch_size=knn_batch_size,
        num_workers=12,
    )

    n_class = knn_train_dataset.get_num_classes()

    knn_val_dataset = KNNDataset(com6_ds, com6_labels, n_packets, 1 - ratio)
    knn_val_loader = DataLoader(
        knn_val_dataset,
        batch_size=knn_batch_size,
        num_workers=12,
    )

    benchmark_model = CLEAN(
        lr_factor=8.0,
        in_channels=2,
        packets=n_packets,
        subcarries=52,
        dataloader_kNN=knn_train_loader,
        num_classes=n_class,
        knn_k=49,
    )

    model_name = CLEAN.__name__.replace("Model", "")

    sub_dir = model_name
    logger = TensorBoardLogger(save_dir="logs/pre-train", name="", sub_dir=sub_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="gpu",
        devices=1,
        default_root_dir="logs/pre-train",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        benchmark=True,
        log_every_n_steps=10,
        precision=32,
    )
    start = time.time()
    trainer.fit(
        benchmark_model,
        train_dataloaders=ssl_loader,
        val_dataloaders=knn_val_loader,
    )
    end = time.time()
    run = {
        "model": model_name,
        "batch_size": ssl_batch_size,
        "epochs": n_epochs,
        "max_accuracy": benchmark_model.max_accuracy,
        "runtime": end - start,
        "gpu_memory_usage": torch.cuda.max_memory_allocated(),
    }
    print(run)
