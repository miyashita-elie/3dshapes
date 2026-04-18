import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from differentiable_datamodule import Differentiable3DShapesDataModule, FactorRangeConfig
from module import CLNFSupervisedModule


def build_datamodule(args):
    num_workers = args.num_workers
    if num_workers != 0:
        print('[clnf_supervised.py] num_workers is forced to 0 in differentiable mode.')
        num_workers = 0

    return Differentiable3DShapesDataModule(
        train_samples_per_epoch=args.train_samples_per_epoch,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        num_workers=num_workers,
        train_resample_each_epoch=not args.disable_train_resample,
        return_grad=True,
        seed=args.seed_data,
        render_device=args.render_device,
        factor_range=FactorRangeConfig(
            size_min=args.size_min,
            size_max=args.size_max,
        ),
    )


def main(args):
    torch.manual_seed(args.seed_model)

    pl_model = CLNFSupervisedModule(
        ckpt_predictor=args.ckpt_predictor,
        ckpt_autoencoder=args.ckpt_autoencoder,
        flow_layers=args.flow_layers,
        flow_hidden_dim=args.flow_hidden_dim,
        scale_map=args.scale_map,
        normalize_generators=args.normalize_generators,
        lr=args.lr,
        sample_num=args.sample_num,
        generator_num=args.generator_num,
        repr_dims=args.repr_dims,
    )

    datamodule = build_datamodule(args)

    wandb_logger = WandbLogger(project=args.project, name=args.run_name)
    wandb_logger.log_hyperparams(dict(
        ckpt_predictor=args.ckpt_predictor,
        ckpt_autoencoder=args.ckpt_autoencoder,
        flow_layers=args.flow_layers,
        flow_hidden_dim=args.flow_hidden_dim,
        scale_map=args.scale_map,
        normalize_generators=args.normalize_generators,
        lr=args.lr,
        batch_size=args.batch_size,
        train_samples_per_epoch=args.train_samples_per_epoch,
        val_samples=args.val_samples,
        size_min=args.size_min,
        size_max=args.size_max,
        seed_model=args.seed_model,
        seed_data=args.seed_data,
    ))

    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir if type(wandb_logger.experiment.dir) is str else args.ckpt_dir,
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        save_top_k=1,
        save_last=True,
        mode='min',
        save_weights_only=False,
        verbose=True,
    )

    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir if type(wandb_logger.experiment.dir) is str else args.ckpt_dir,
        filename='epoch{epoch:04d}',
        every_n_epochs=args.periodic_save_every,
        save_top_k=-1,
        save_last=False,
        save_weights_only=False,
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, periodic_checkpoint_callback],
        enable_checkpointing=True,
    )
    trainer.fit(pl_model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('ckpt_predictor', type=str, help='Path to pretrained predictor checkpoint')
    parser.add_argument('ckpt_autoencoder', type=str, help='Path to pretrained autoencoder checkpoint')

    parser.add_argument('--flow_layers', type=int, default=24, help='Number of layers in normalizing flow')
    parser.add_argument('--flow_hidden_dim', type=int, default=192, help='Hidden dimension of flow networks')
    parser.add_argument('--scale_map', type=str, default='exp_clamp', help='Scale map for flow (exp, exp_clamp)')
    parser.add_argument('--normalize_generators', action='store_true', help='Normalize generators during training')

    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=2000, help='Max training epochs')
    parser.add_argument('--val_interval', type=int, default=10, help='Validation interval (epochs)')

    parser.add_argument('--sample_num', type=int, default=64, help='Number of generated images for validation plotting')
    parser.add_argument('--generator_num', type=int, default=16, help='Number of generators for transform plotting')
    parser.add_argument('--repr_dims', type=int, nargs='+', default=[7, 8, 9, 12], help='Representation dimensions to analyze')

    parser.add_argument('--train_samples_per_epoch', type=int, default=480000, help='Number of train samples per epoch')
    parser.add_argument('--val_samples', type=int, default=24000, help='Number of validation samples')
    parser.add_argument('--disable_train_resample', action='store_true', help='Disable train-factor resampling each epoch')
    parser.add_argument('--render_device', type=str, default=None, help="Render device: e.g. 'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument('--size_min', type=float, default=0.75, help='Minimum object size factor')
    parser.add_argument('--size_max', type=float, default=1.25, help='Maximum object size factor')

    parser.add_argument('--seed_model', type=int, default=0, help='Model initialization seed')
    parser.add_argument('--seed_data', type=int, default=0, help='Data sampling seed')

    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers (forced to 0)')
    parser.add_argument('--project', type=str, default='3dshapes-clnf-supervised', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='clnf_supervised', help='wandb run name')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/clnf_supervised', help='Checkpoint directory')
    parser.add_argument('--periodic_save_every', type=int, default=100, help='Save periodic checkpoints every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--strategy', type=str, default='auto', help='Distributed training strategy')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training')
    parser.add_argument('--devices', type=str, default='auto', help='Number of devices (GPUs/CPUs) per node')

    args = parser.parse_args()
    main(args)
