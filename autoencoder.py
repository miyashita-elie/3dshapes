import numpy as np
import h5py
import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from module import AutoencoderModule
from dataset import Dataset3DShapes
from differentiable_datamodule import Differentiable3DShapesDataModule, FactorRangeConfig


def _build_h5_dataloaders(args):
    dataset = h5py.File('3dshapes.h5', 'r')
    print(dataset.keys())
    images = dataset['images'][:]
    images = images.reshape(10, 10, 10, 8, 4, 15, 64, 64, 3)

    s = [slice(None)] * 6
    factor_dict = {
        'floor_hue': 0,
        'wall_hue': 1,
        'object_hue': 2,
        'scale': 3,
        'shape': 4,
        'orientation': 5,
    }
    for factor in args.removed_factors:
        idx = factor_dict[factor]
        s[idx] = 0
    s = tuple(s)

    images = images[s]
    images = images.reshape(-1, 64, 64, 3)
    n_samples = images.shape[0]

    # データ分割（1:7）
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(n_samples / 8)
    val_idx, train_idx = indices[:split], indices[split:]

    train_data = Dataset3DShapes(images=images, indices=train_idx)
    val_data = Dataset3DShapes(images=images, indices=val_idx)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    return train_loader, val_loader


def _build_differentiable_datamodule(args):
    num_workers = args.num_workers
    if num_workers != 0:
        print('[autoencoder.py] num_workers is forced to 0 in differentiable mode.')
        num_workers = 0

    if args.removed_factors:
        print('[autoencoder.py] removed_factors is ignored in differentiable mode.')

    return Differentiable3DShapesDataModule(
        train_samples_per_epoch=args.train_samples_per_epoch,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        num_workers=num_workers,
        return_grad=False,
        seed=args.seed_data,
        render_device=args.render_device,
        factor_range=FactorRangeConfig(
            size_min=args.size_min,
            size_max=args.size_max,
        ),
    )


def main(args):
    # Set up model
    torch.manual_seed(args.seed_model)

    pl_model = AutoencoderModule(
        num_post_layers=args.num_post_layers,
        latent_dim=args.latent_dim,
        sample_num=args.sample_num,
    )

    # wandb logger
    wandb_logger = WandbLogger(project=args.project, name=args.run_name)

    # ModelCheckpointコールバック
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir if type(wandb_logger.experiment.dir) is str else args.ckpt_dir,
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        save_weights_only=False,
        verbose=True,
    )

    trainer = Trainer(
        max_steps=args.max_steps,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    if args.data_mode == 'h5':
        train_loader, val_loader = _build_h5_dataloaders(args)
        trainer.fit(pl_model, train_loader, val_loader, ckpt_path=args.resume)
    else:
        datamodule = _build_differentiable_datamodule(args)
        trainer.fit(pl_model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Backward-compatible data mode selector.
    parser.add_argument('--data_mode', type=str, default='h5', choices=['h5', 'differentiable'], help='Data pipeline mode')

    # Existing arguments (kept for compatibility)
    parser.add_argument('--removed_factors', type=str, nargs='+', default=[], help='Factors to remove from the dataset. Options: floor_hue, wall_hue, object_hue, scale, shape, orientation')
    parser.add_argument('--num_post_layers', type=int, default=3, help='Number of post-processing layers in the decoder')
    parser.add_argument('--latent_dim', type=int, default=12, help='Latent dimension')
    parser.add_argument('--batch_size', type=int, default=16000, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=120000, help='Max training steps')
    parser.add_argument('--val_interval', type=int, default=50, help='Validation interval in steps')
    parser.add_argument('--sample_num', type=int, default=64, help='Number of generated images per validation')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of DataLoader workers')
    parser.add_argument('--project', type=str, default='3dshapes-autoencoder', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='autoencoder', help='wandb run name')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/autoencoder', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--strategy', type=str, default='auto', help='Distributed training strategy (ddp, ddp_spawn, etc)')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training')
    parser.add_argument('--devices', type=str, default='auto', help='Number of devices (GPUs/CPUs) per node')

    # Differentiable data mode arguments
    parser.add_argument('--train_samples_per_epoch', type=int, default=480000, help='Number of train samples per epoch (differentiable mode)')
    parser.add_argument('--val_samples', type=int, default=24000, help='Number of val samples (differentiable mode)')
    parser.add_argument('--render_device', type=str, default=None, help="Render device: e.g. 'cuda', 'cuda:0', or 'cpu' (differentiable mode)")
    parser.add_argument('--size_min', type=float, default=0.75, help='Minimum object size factor (differentiable mode)')
    parser.add_argument('--size_max', type=float, default=1.25, help='Maximum object size factor (differentiable mode)')
    parser.add_argument('--seed_model', type=int, default=0, help='Model initialization seed')
    parser.add_argument('--seed_data', type=int, default=0, help='Data sampling seed (differentiable mode)')

    args = parser.parse_args()
    main(args)
