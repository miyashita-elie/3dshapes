import numpy as np
import h5py
import argparse
import os

import torch
import torchvision as tv
import normflows as nf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from module import CLNFModule
from dataset import Dataset3DShapes


def main(args):
    # load dataset
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
        'orientation': 5
    }
    for factor in args.removed_factors:
        idx = factor_dict[factor]
        s[idx] = 0
    for factor in args.cut_factors:
        idx = factor_dict[factor]
        s[idx] = slice(0, 7)  # 最初の7つの値を使用
    s = tuple(s)
    images = images[s]
    images = images.reshape(-1, 64, 64, 3)
    n_samples = images.shape[0]
    sample_image = images[0]

    np.random.seed(42)
    torch.manual_seed(0)
    
    # LightningModule化
    pl_model = CLNFModule(
        ckpt_predictor=args.ckpt_predictor,
        ckpt_autoencoder=args.ckpt_autoencoder,
        flow_layers=args.flow_layers,
        flow_hidden_dim=args.flow_hidden_dim,
        hom_layers=args.hom_layers,
        hom_hidden_dim=args.hom_hidden_dim,
        scale_map=args.scale_map,
        num_bases_sym=args.num_bases_sym,
        num_bases_null=args.num_bases_null,
        eps_p_sym=args.eps_p_sym,
        eps_q_sym=args.eps_q_sym,
        eps_p_null=args.eps_p_null,
        eps_q_null=args.eps_q_null,
        normalize_generators=args.normalize_generators,
        normalize_precision=args.normalize_precision,
        rescale_eps=args.rescale_eps,
        lr=args.lr,
        sample_image=sample_image,
        sample_num=args.sample_num,
        generator_num=args.generator_num,
        repr_dims=args.repr_dims,
        predicted_factors=args.predicted_factors,
    )

    # DataLoader
    indices = np.arange(n_samples)
    train_data = Dataset3DShapes(images=images, indices=indices)
    val_data = Dataset3DShapes(images=images, indices=indices)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # wandb logger
    wandb_logger = WandbLogger(project=args.project, name=args.run_name)

    wandb_logger.log_hyperparams(dict(
        ckpt_predictor=args.ckpt_predictor,
        ckpt_autoencoder=args.ckpt_autoencoder,
        flow_layers=args.flow_layers,
        flow_hidden_dim=args.flow_hidden_dim,
        hom_layers=args.hom_layers,
        hom_hidden_dim=args.hom_hidden_dim,
        scale_map=args.scale_map,
        num_bases_sym=args.num_bases_sym,
        num_bases_null=args.num_bases_null,
        eps_p_sym=args.eps_p_sym,
        eps_q_sym=args.eps_q_sym,
        eps_p_null=args.eps_p_null,
        eps_q_null=args.eps_q_null,
        normalize_generators=args.normalize_generators,
        normalize_precision=args.normalize_precision,
        rescale_eps=args.rescale_eps,
        lr=args.lr,
        sample_num=args.sample_num,
        generator_num=args.generator_num,
        repr_dims=args.repr_dims,
    ))

    # ModelCheckpointコールバック
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir if type(wandb_logger.experiment.dir) is str else args.ckpt_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        mode="min",
        save_weights_only=False,
        verbose=True
    )

    # 500エポックごとに保存するModelCheckpointコールバック
    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir if type(wandb_logger.experiment.dir) is str else args.ckpt_dir,
        filename="epoch{epoch:04d}",
        every_n_epochs=500,
        save_top_k=-1,
        save_last=False,
        save_weights_only=False,
        verbose=True
    )

    if args.backend is not None:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(process_group_backend=args.backend, find_unused_parameters=True)
    else:
        strategy = args.strategy

    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        check_val_every_n_epoch=args.val_interval,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, periodic_checkpoint_callback],
        enable_checkpointing=True
    )
    trainer.fit(pl_model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--removed_factors', type=str, nargs='+', default=[], help='Factors to remove from the dataset. Options: floor_hue, wall_hue, object_hue, scale, shape, orientation')
    parser.add_argument('--predicted_factors', type=str, nargs='+', default=['scale', 'shape'], help='Factors to predict from the dataset. Options: floor_hue, wall_hue, object_hue, scale, shape, orientation')
    parser.add_argument('--cut_factors', type=str, nargs='+', default=[], help='Factors to cut from the dataset. Options: floor_hue, wall_hue, object_hue, scale, shape, orientation')

    parser.add_argument('ckpt_predictor', type=str, help='Path to pretrained predictor checkpoint')
    parser.add_argument('ckpt_autoencoder', type=str, help='Path to pretrained autoencoder checkpoint')
    # flow parameters
    parser.add_argument('--flow_layers', type=int, default=24, help='Number of layers in normalizing flow')
    parser.add_argument('--flow_hidden_dim', type=int, default=192, help='Hidden dimension of flow networks')
    parser.add_argument('--hom_layers', type=int, default=0, help='Number of layers in homomorphic flow')
    parser.add_argument('--hom_hidden_dim', type=int, default=192, help='Hidden dimension of homomorphic flow networks')
    parser.add_argument('--scale_map', type=str, default='exp_clamp', help='Scale map for flow (exp, exp_clamp)')
    # symmetry parameters
    parser.add_argument('--num_bases_sym', type=int, default=None, help='Number of bases for symmetric part')
    parser.add_argument('--num_bases_null', type=int, default=None, help='Number of bases for null part')
    parser.add_argument('--eps_p_sym', type=float, default=1e-3, help='Epsilon p for symmetric part')
    parser.add_argument('--eps_q_sym', type=float, default=1e-1, help='Epsilon q for symmetric part')
    parser.add_argument('--eps_p_null', type=float, default=1e-3, help='Epsilon p for null part')
    parser.add_argument('--eps_q_null', type=float, default=1e-1, help='Epsilon q for null part')
    parser.add_argument('--normalize_generators', action='store_true', help='Whether to normalize generators during training')
    parser.add_argument('--normalize_precision', action='store_true', help='Whether to normalize precision matrices during training')
    parser.add_argument('--rescale_eps', action='store_true', help='Whether to rescale epsilons during training')
    # plot parameters
    parser.add_argument('--sample_num', type=int, default=64, help='Number of generated images per validation')
    parser.add_argument('--generator_num', type=int, default=64, help='Number of generators to estimate')
    parser.add_argument('--repr_dims', type=int, nargs='+', default=[7,8,9,24], help='Representation dimensions to analyze')
    # training parameters
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=5000, help='Max training steps')
    parser.add_argument('--val_interval', type=int, default=10, help='Validation interval in steps')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--project', type=str, default='3dshapes-cotflow', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='clnf_run', help='wandb run name')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/clnf', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--strategy', type=str, default='auto', help='Distributed training strategy (ddp, ddp_spawn, etc)')
    parser.add_argument('--backend', type=str, default=None, help='Distributed backend (nccl, gloo, etc)')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for distributed training')
    parser.add_argument('--devices', type=str, default='auto', help='Number of devices (GPUs/CPUs) per node')
    args = parser.parse_args()
    main(args)
