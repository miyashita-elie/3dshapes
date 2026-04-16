import torch
import pytorch_lightning as pl
import torchvision as tv
import wandb
import normflows as nf
import math
from typing import Any
from matplotlib import pyplot as plt

class CustomError(Exception):
    pass

class Autoencoder(torch.nn.Module):
    def __init__(self, num_post_layers=2, latent_dim=24):
        super().__init__()
        self.latent_dim = latent_dim
        layers = [
            torch.nn.Conv2d(3, 3*2, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*2),
            torch.nn.Conv2d(3*2, 3*4, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*4),
            torch.nn.Conv2d(3*4, 3*8, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*8),
            torch.nn.Conv2d(3*8, 3*16, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*16),
            torch.nn.Conv2d(3*16, 3*32, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*32),
            torch.nn.Conv2d(3*32, 3*64, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*64),
        ]

        current_channels = 3*64
        for _ in range(num_post_layers):
            layers.extend([
                torch.nn.Conv2d(current_channels, current_channels // 2, kernel_size=1),
                torch.nn.Softplus(),
                torch.nn.BatchNorm2d(current_channels // 2),
            ])
            current_channels //= 2

        layers.append(torch.nn.Conv2d(current_channels, latent_dim, kernel_size=1))
        layers.append(torch.nn.BatchNorm2d(latent_dim, affine=False))

        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        layers.append(torch.nn.ConvTranspose2d(latent_dim, current_channels, kernel_size=1))

        for _ in range(num_post_layers):
            layers.extend([
                torch.nn.Softplus(),
                torch.nn.BatchNorm2d(current_channels),
                torch.nn.ConvTranspose2d(current_channels, current_channels * 2, kernel_size=1),
            ])
            current_channels *= 2

        layers.extend([
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*64),
            torch.nn.ConvTranspose2d(3*64, 3*32, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*32),
            torch.nn.ConvTranspose2d(3*32, 3*16, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*16),
            torch.nn.ConvTranspose2d(3*16, 3*8, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*8),
            torch.nn.ConvTranspose2d(3*8, 3*4, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*4),
            torch.nn.ConvTranspose2d(3*4, 3*2, kernel_size=2, stride=2),
            torch.nn.Softplus(),
            torch.nn.BatchNorm2d(3*2),
            torch.nn.ConvTranspose2d(3*2, 3, kernel_size=2, stride=2),
            torch.nn.Sigmoid(),
        ])

        self.decoder = torch.nn.Sequential(*layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


class AutoencoderModule(pl.LightningModule):
    def __init__(
        self, 
        num_post_layers=2,
        latent_dim=24,
        sample_num=64
    ):
        super().__init__()
        self.model = Autoencoder(num_post_layers=num_post_layers, latent_dim=latent_dim)
        self.recon_imgs = None
        self.sample_num = sample_num
            
    def on_validation_epoch_end(self):
        # 検証終了時に画像生成しwandbに記録
        if self.recon_imgs is None:
            return
        with torch.no_grad():
            img = self.recon_imgs[:self.sample_num]
            grid = tv.utils.make_grid(img.cpu(), nrow=8)
            wandb_logger = self.logger
            if hasattr(wandb_logger, "experiment"):
                wandb_logger.experiment.log({f"images/sample": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})
            self.recon_imgs = None

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)

    def forward(self, x):
        return self.model(x)

    def _extract_images(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            if "image" not in batch:
                raise KeyError("Dictionary batch must contain key 'image' for autoencoder training.")
            return batch["image"]
        return batch

    def training_step(self, batch, batch_idx):
        imgs = self._extract_images(batch)
        recon = self.model(imgs)[1]
        loss = torch.nn.functional.mse_loss(recon, imgs)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = self._extract_images(batch)
        recon = self.model(imgs)[1]
        if self.recon_imgs is None:
            self.recon_imgs = recon.clamp(0, 1).cpu()
        loss = torch.nn.functional.mse_loss(recon, imgs)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=1e-3)
        return optimizer


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3*2, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*2, 64, 64]),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(3*2, 3*4, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*4, 32, 32]),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(3*4, 3*8, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*8, 16, 16]),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(3*8, 3*16, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*16, 8, 8]),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(3*16, 3*32, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*32, 4, 4]),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(3*32, 3*64, kernel_size=3, padding=1),
            torch.nn.Softplus(),
            torch.nn.LayerNorm([3*64, 2, 2]),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten()
        )

        self.color_predictor = torch.nn.Sequential(
            torch.nn.Linear(3*64, 128),
            torch.nn.Softplus(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 6)
        )

        self.scale_predictor = torch.nn.Sequential(
            torch.nn.Linear(3*64, 128),
            torch.nn.Softplus(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 1)
        )

        self.shape_predictor = torch.nn.Sequential(
            torch.nn.Linear(3*64, 128),
            torch.nn.Softplus(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 4)
        )

        self.orientation_predictor = torch.nn.Sequential(
            torch.nn.Linear(3*64, 128),
            torch.nn.Softplus(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.feature_extractor.forward(x)
        color = self.color_predictor.forward(features).reshape(-1, 3, 2)
        color = color / torch.norm(color, dim=-1, keepdim=True)
        scale = self.scale_predictor.forward(features)
        shape = self.shape_predictor.forward(features).softmax(dim=-1)
        orientation = self.orientation_predictor.forward(features).squeeze(-1)
        return color, scale, shape, orientation


class PredictorModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Predictor()
            
    def forward(self, x):
        return self.model(x)
    
    def criterion(self, preds, targets):
        color_pred, scale_pred, shape_pred, orientation_pred = preds
        color_target, scale_target, shape_target, orientation_target = targets

        color_loss = torch.nn.functional.mse_loss(color_pred, color_target)
        scale_loss = torch.nn.functional.mse_loss(scale_pred, scale_target)
        shape_loss = torch.nn.functional.cross_entropy(shape_pred.add(1).log(), shape_target)
        orientation_loss = torch.nn.functional.mse_loss(orientation_pred, orientation_target)

        shape_acc = (shape_pred.argmax(dim=-1) == shape_target).float().mean()

        total_loss = color_loss + scale_loss + shape_loss + orientation_loss
        return total_loss, color_loss, scale_loss, shape_loss, orientation_loss, shape_acc

    def training_step(self, batch, batch_idx):
        imgs, color_target, scale_target, shape_target, orientation_target = batch
        color, scale, shape, orientation = self.model(imgs)
        loss, color_loss, scale_loss, shape_loss, orientation_loss, shape_acc = self.criterion(
            (color, scale, shape, orientation),
            (color_target, scale_target, shape_target, orientation_target)
        )
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_color_loss', color_loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_scale_loss', scale_loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_shape_loss', shape_loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_shape_acc', shape_acc, on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_orientation_loss', orientation_loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, color_target, scale_target, shape_target, orientation_target = batch
        color, scale, shape, orientation = self.model(imgs)
        loss, color_loss, scale_loss, shape_loss, orientation_loss, shape_acc = self.criterion(
            (color, scale, shape, orientation),
            (color_target, scale_target, shape_target, orientation_target)
        )
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_color_loss', color_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_scale_loss', scale_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_shape_loss', shape_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_shape_acc', shape_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_orientation_loss', orientation_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss 

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=1e-3, weight_decay=1e-1)
        return optimizer


class NFModule(pl.LightningModule):
    def __init__(self, sample_num=64, beta=10.0):
        super().__init__()
        self.sample_num = sample_num
        self.beta = beta

        num_layers = 12
        input_dim = 128
        hidden_dim = 128
        half_dim = input_dim // 2

        self.autoencoder = Autoencoder(input_dim)

        base = nf.distributions.base.DiagGaussian(input_dim, trainable=False)

        flows = []
        for i in range(num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([half_dim, hidden_dim, hidden_dim, input_dim], init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(input_dim, mode='swap'))

        self.flow = nf.NormalizingFlow(base, flows)

    def on_validation_epoch_end(self):
        # 検証終了時に画像生成しwandbに記録
        with torch.no_grad():
            z = self.flow.sample(self.sample_num)[0]
            z = z.view(self.sample_num, -1, 1, 1)
            img = self.autoencoder.decode(z)
            img = img.clamp(0, 1)
            grid = tv.utils.make_grid(img.cpu(), nrow=8)
            wandb_logger = self.logger
            if hasattr(wandb_logger, "experiment"):
                wandb_logger.experiment.log({f"val_generated/sample": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})

    def encode(self, x):
        z = self.autoencoder.encode(x)
        z = z.reshape(z.size(0), -1)
        return self.flow.inverse(z)
    
    def decode(self, z):
        z = self.flow.forward(z)
        z = z.reshape(z.size(0), -1, 1, 1)
        return self.autoencoder.decode(z)
    
    def sample(self, num_samples):
        z = self.flow.sample(num_samples)[0]
        z = z.reshape(num_samples, -1, 1, 1)
        return self.autoencoder.decode(z)

    def forward(self, x):
        z, recon = self.autoencoder(x)
        z = z.reshape(z.size(0), -1)
        return self.flow(z)

    def training_step(self, batch, batch_idx):
        z, recon = self.autoencoder(batch)
        z = z.reshape(z.size(0), -1)
        nll = self.flow.forward_kld(z.detach())
        recons = torch.nn.functional.mse_loss(recon, batch)
        loss = self.beta * recons + nll
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_nll', nll, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recons', recons, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z, recon = self.autoencoder(batch)
        z = z.reshape(z.size(0), -1)
        nll = self.flow.forward_kld(z)
        recons = torch.nn.functional.mse_loss(recon, batch)
        loss = self.beta * recons + nll
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_nll', nll, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recons', recons, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(list(self.autoencoder.parameters()) + list(self.flow.parameters()), lr=1e-3, weight_decay=1e-5)
        return optimizer


class CLNF(torch.nn.Module):
    def __init__(
        self, 
        ckpt_predictor,
        ckpt_autoencoder,
        flow_layers=24,
        flow_hidden_dim=192,
        hom_layers=24,
        hom_hidden_dim=192,
        scale_map="exp_clamp",
        num_bases_sym=None,
        num_bases_null=None,
        eps_p_sym=1e-3,
        eps_q_sym=1e-1,
        eps_p_null=1e-3,
        eps_q_null=1e-1,
        normalize_generators=False,
        normalize_precision=False,
        rescale_eps=False,
        fix_w_sym_to_commutative_rotation_basis=False,
        predicted_factors=['scale', 'shape'],
    ):
        super().__init__()

        self.normalize_generators = normalize_generators
        self.normalize_precision = normalize_precision
        self.rescale_eps = rescale_eps
        self.fix_w_sym_to_commutative_rotation_basis = fix_w_sym_to_commutative_rotation_basis

        self.predictor = PredictorModule.load_from_checkpoint(ckpt_predictor).model.eval()
        self.autoencoder = AutoencoderModule.load_from_checkpoint(ckpt_autoencoder).model.eval()

        self.latent_dim = self.autoencoder.latent_dim
        self.num_bases_sym = num_bases_sym
        self.num_bases_null = num_bases_null

        input_dim = self.latent_dim
        hidden_dim = flow_hidden_dim
        half_dim = input_dim // 2

        base = nf.distributions.base.DiagGaussian(input_dim, trainable=False)

        flows = []
        for i in range(flow_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([half_dim, hidden_dim, hidden_dim, input_dim], init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map=scale_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(input_dim, mode='swap'))

        self.flow = nf.NormalizingFlow(base, flows)


        if hom_layers > 0:
            flows = []
            for i in range(hom_layers):
                # Neural network with two hidden layers having 64 units each
                # Last layer is initialized by zeros making training more stable
                param_map = nf.nets.MLP([half_dim, hom_hidden_dim, hom_hidden_dim, input_dim], init_zeros=True)
                # Add flow layer
                flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map=scale_map))
                # Swap dimensions
                flows.append(nf.flows.Permute(input_dim, mode='swap'))

            self.hom = nf.NormalizingFlow(base, flows)
        else:
            self.hom = None

        if num_bases_sym is None:
            self.W_sym = None
        else:
            if self.fix_w_sym_to_commutative_rotation_basis:
                self.register_buffer(
                    'W_sym',
                    self._commutative_rotation_basis(input_dim, num_bases_sym),
                )
            else:
                self.W_sym = torch.nn.Parameter(torch.randn(num_bases_sym, input_dim, input_dim) * (1.0 / math.sqrt(input_dim)))
            self.register_buffer('eps_p_sym', torch.tensor(eps_p_sym))
            self.register_buffer('eps_q_sym', torch.tensor(eps_q_sym))
        if num_bases_null is None:
            self.W_null = None
        else:
            self.W_null = torch.nn.Parameter(torch.randn(num_bases_null, input_dim, input_dim) * (1.0 / math.sqrt(input_dim)))
            self.register_buffer('eps_p_null', torch.tensor(eps_p_null))
            self.register_buffer('eps_q_null', torch.tensor(eps_q_null))

        self.register_buffer('var_sym', torch.tensor(-1.0))
        self.register_buffer('var_null', torch.tensor(-1.0))

        self.predicted_factors = predicted_factors
        self.predict_mask = torch.zeros(12, dtype=torch.float32)
        for factor in predicted_factors:
            if factor == 'floor_hue':
                self.predict_mask[0:2] = 1.0
            elif factor == 'wall_hue':
                self.predict_mask[2:4] = 1.0
            elif factor == 'object_hue':
                self.predict_mask[4:6] = 1.0
            elif factor == 'scale':
                self.predict_mask[6:7] = 1.0
            elif factor == 'shape':
                self.predict_mask[7:11] = 1.0
            elif factor == 'orientation':
                self.predict_mask[11:12] = 1.0

    def _commutative_rotation_basis(self, latent_dim, num_bases):
        basis = []
        for k in range(num_bases):
            mat = torch.zeros(latent_dim, latent_dim)
            i = 2 * k
            j = 2 * k + 1
            if i < latent_dim and j < latent_dim:
                mat[i, j] = 1.0
                basis.append(mat)
        return torch.stack(basis, dim=0)

    def parameters(self, recurse = True):
        yield from self.flow.parameters(recurse)
        if self.hom is not None:
            yield from self.hom.parameters(recurse)
        if self.W_sym is not None:
            yield self.W_sym
        if self.W_null is not None:
            yield self.W_null

    def encode(self, x):
        y = self.autoencoder.encode(x)
        y = y.reshape(y.size(0), -1)
        z = self.flow.inverse(y)
        if self.hom is not None:
            z = self.hom.inverse(z)
        return z

    def decode(self, z):
        if self.hom is not None:
            z = self.hom.forward(z)
        y = self.flow.forward(z)
        y = y.reshape(y.size(0), -1, 1, 1)
        x = self.autoencoder.decode(y)
        return x
    
    def _predict(self, x: torch.Tensor):
        # x: (3, 64, 64)
        # Returns: (output_dim,)
        x = x.unsqueeze(0)
        color, scale, shape, orientation = self.predictor.forward(x)
        color = color.reshape(1, -1)
        orientation = orientation.unsqueeze(-1)
        out = torch.cat([color, scale, shape, orientation], dim=-1).squeeze(0)
        out = out * self.predict_mask.to(out.device)
        return out

    def _decode(self, y: torch.Tensor):
        # y: (input_dim,)
        # Returns: (output_dim,)
        y = y.reshape(1, -1, 1, 1)
        x = self.autoencoder.decode(y)
        return x.squeeze(0)
    
    def _forward_flow(self, z: torch.Tensor):
        # z: (input_dim,)
        # Returns: (output_dim,)
        z = z.unsqueeze(0)
        if self.hom is not None:
            z = self.hom.forward(z)
        y = self.flow.forward(z)
        return y.squeeze(0)
    
    def sample_cotangent(self, x: torch.Tensor, cv: torch.Tensor):
        def _sample_cotangent_single(x: torch.Tensor, cv: torch.Tensor):
            # x: (3, 64, 64)
            # cv: (output_dim,)
            # Returns: (input_dim,)

            _, vjp_fn = torch.func.vjp(self._predict, x)

            cv = vjp_fn(cv)[0]

            return cv

        cv = torch.func.vmap(_sample_cotangent_single)(x, cv)

        return cv
    
    def encode_cotangent(self, y: torch.Tensor, cv: torch.Tensor):
        def _encode_cotangent_single(y: torch.Tensor, cv: torch.Tensor):
            # y: (input_dim,)
            # cv: (output_dim,)
            # Returns: (input_dim,)

            _, vjp_fn = torch.func.vjp(self._decode, y)

            cv = vjp_fn(cv)[0]

            return cv

        cv = torch.func.vmap(_encode_cotangent_single)(y, cv)

        return cv

    def pullback_cotangent(self, z: torch.Tensor, cv: torch.Tensor):
        def _pullback_cotangent_single(z: torch.Tensor, cv: torch.Tensor):
            _, vjp_fn = torch.func.vjp(self._forward_flow, z)

            cv = vjp_fn(cv)[0]

            return cv

        cv = torch.func.vmap(_pullback_cotangent_single)(z, cv)

        return cv

    def log_prob(
        self, 
        cv: torch.Tensor, 
        J: torch.Tensor, 
        eps_p: torch.Tensor,
        eps_q: torch.Tensor,
    ):
        """
        Args:
            cv: (batch_size, input_dim)
            J: (batch_size, num_bases, input_dim)
        Returns:
            log_prob: (batch_size,)
        """

        input_dim = J.size(-1)
        num_bases = J.size(-2)

        S_q = torch.einsum('bin,bim->bnm', J, J)                        # (batch_size, input_dim, input_dim)
        if self.normalize_precision:
            S_q = S_q / num_bases

        I_q = torch.eye(input_dim, device=J.device)                       # (input_dim, input_dim)
        if self.rescale_eps:
            norm = S_q.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)
            H = S_q + eps_q * norm.unsqueeze(-1).unsqueeze(-1) * I_q
        else:
            H = S_q + eps_q * I_q
        M = eps_p + torch.einsum('bi,bi->b', cv, cv)                   # (batch_size,)

        cv = cv + eps_p.sqrt() * torch.randn_like(cv)
        trace = torch.einsum('bi,bij,bj->b', cv, H, cv)                # (batch_size,)

        try:
            L_H = torch.linalg.cholesky(H)
        except Exception:
            raise CustomError("Cholesky decomposition failed in log_prob computation.")

        logdet = 2 * torch.log(torch.diagonal(L_H, dim1=-2, dim2=-1)).sum(-1)
        logdet = logdet + torch.log(M) + (input_dim - 1) * torch.log(eps_p)  # (batch_size,)

        log_prob = 0.5 * (logdet - trace - input_dim * math.log(2 * math.pi))  # (batch_size,)

        return log_prob

    @torch.no_grad()
    def sample(self, num_samples):
        z = self.flow.sample(num_samples)[0]
        z = z.reshape(num_samples, -1, 1, 1)
        x = self.autoencoder.decode(z)
        return x

    def forward(self, img: torch.Tensor):
        y = self.autoencoder.encode(img)
        x = self.autoencoder.decode(y)

        y = y.detach().reshape(y.size(0), -1)
        z, logdet = self.flow.inverse_and_log_det(y)
        log_prob_data = self.flow.q0.log_prob(z) + logdet  # (B,)
        if self.hom is not None:
            z = self.hom.inverse(z)

        cv = torch.randn(x.size(0), 12, device=x.device)  # (B, output_dim)
        cv = self.sample_cotangent(x.detach(), cv)  # (B, input_dim)
        cv_sym = self.encode_cotangent(y, cv.detach())  # (B, input_dim)
        cv_sym = cv_sym.detach()
        cv = torch.randn_like(x)
        cv_null = self.encode_cotangent(y, cv)  # (B, input_dim)
        cv_null = cv_null.detach()

        if self.training:
            with torch.no_grad():
                var_sym = cv_sym.square().mean()
                if self.var_sym.item() < 0:
                    self.var_sym.copy_(var_sym)
                else:
                    self.var_sym.mul_(0.9).add_(0.1 * var_sym)

                var_null = cv_null.square().mean()
                if self.var_null.item() < 0:
                    self.var_null.copy_(var_null)
                else:
                    self.var_null.mul_(0.9).add_(0.1 * var_null)
        
        cv_sym = cv_sym / self.var_sym.clamp_min(1e-6).sqrt()
        cv_sym = self.pullback_cotangent(z, cv_sym)  # (B, input_dim)

        cv_null = cv_null / self.var_null.clamp_min(1e-6).sqrt()
        cv_null = self.pullback_cotangent(z, cv_null)  # (B, input_dim)

        output_dict = dict(z=z, x=x, cv_sym=cv_sym, cv_null=cv_null, log_prob_data=log_prob_data)

        if self.W_sym is not None:
            L = (self.W_sym - self.W_sym.mT) / 2  # (num_bases, input_dim, input_dim)
            if self.normalize_generators:
                L = L / L.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6).sqrt()
                L = L / L.size(-1) ** 0.5
            J_sym = torch.einsum('bi,mji->bmj', z, L)  # (B, num_bases, input_dim)
            log_prob_sym = self.log_prob(cv_sym, J_sym, self.eps_p_sym, self.eps_q_sym)  # (B,)

            output_dict.update(log_prob_sym=log_prob_sym)

            if self.hom is not None:
                y_sample, _ = self.flow.sample(z.size(0))
                logp1 = self.flow.log_prob(y_sample)
                kl_sym = logp1
                z_sample, logdet1 = self.flow.inverse_and_log_det(y_sample)
                kl_sym = kl_sym - logdet1
                z_sample, logdet2 = self.hom.inverse_and_log_det(z_sample)
                kl_sym = kl_sym - logdet2
                eps = torch.randn(z.size(0), self.num_bases_sym, device=z.device) / self.num_bases_sym
                A = torch.eye(self.latent_dim, device=z.device) + torch.einsum('bm,mji->bji', eps, L)
                z_sample = torch.einsum('bji,bi->bj', A, z_sample)
                logdet3 = A.logdet()
                kl_sym = kl_sym - logdet3
                z_sample, logdet4 = self.hom.forward_and_log_det(z_sample)
                kl_sym = kl_sym - logdet4
                y_sample, logdet5 = self.flow.forward_and_log_det(z_sample)
                kl_sym = kl_sym - logdet5
                logp2 = self.flow.log_prob(y_sample)
                kl_sym = kl_sym - logp2

                kl_sym = kl_sym.clamp(min=0.0, max=10.0)

                output_dict.update(kl_sym=kl_sym)

        if self.W_null is not None:
            L = (self.W_null - self.W_null.mT) / 2  # (num_bases, input_dim, input_dim)
            if self.normalize_generators:
                L = L / L.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6).sqrt()
                L = L / L.size(-1) ** 0.5
            J_null = torch.einsum('bi,mji->bmj', z, L)  # (B, num_null, input_dim)

            log_prob_null = self.log_prob(cv_null, J_null, self.eps_p_null, self.eps_q_null)  # (B,)

            output_dict.update(log_prob_null=log_prob_null)

            if self.hom is not None:
                y_sample, _ = self.flow.sample(z.size(0))
                logp1 = self.flow.log_prob(y_sample)
                kl_null = logp1
                z_sample, logdet1 = self.flow.inverse_and_log_det(y_sample)
                kl_null = kl_null - logdet1
                z_sample, logdet2 = self.hom.inverse_and_log_det(z_sample)
                kl_null = kl_null - logdet2
                eps = torch.randn(z.size(0), self.num_bases_null, device=z.device) / self.num_bases_null
                A = torch.eye(self.latent_dim, device=z.device) + torch.einsum('bm,mji->bji', eps, L)
                z_sample = torch.einsum('bji,bi->bj', A, z_sample)
                logdet3 = A.logdet()
                kl_null = kl_null - logdet3
                z_sample, logdet4 = self.hom.forward_and_log_det(z_sample)
                kl_null = kl_null - logdet4
                y_sample, logdet5 = self.flow.forward_and_log_det(z_sample)
                kl_null = kl_null - logdet5
                logp2 = self.flow.log_prob(y_sample)
                kl_null = kl_null - logp2

                kl_null = kl_null.clamp(min=0.0, max=10.0)

                output_dict.update(kl_null=kl_null)


        return output_dict


class CLNFModule(pl.LightningModule):
    def __init__(
        self, 
        ckpt_predictor: str,
        ckpt_autoencoder: str,
        flow_layers=24,
        flow_hidden_dim=192,
        hom_layers=0,
        hom_hidden_dim=192,
        num_bases_sym=None,
        num_bases_null=None,
        scale_map="exp_clamp",
        eps_p_sym=1e-3,
        eps_q_sym=1e-1,
        eps_p_null=1e-3,
        eps_q_null=1e-1,
        normalize_generators=False,
        normalize_precision=False,
        rescale_eps=False,
        fix_w_sym_to_commutative_rotation_basis=False,
        lr=1e-3,
        sample_image: torch.Tensor=None,
        sample_num=64,
        generator_num=16,
        repr_dims=[7, 8, 9, 12],
        predicted_factors=['scale', 'shape'],
    ):
        super().__init__()

        self.model = CLNF(
            ckpt_predictor,
            ckpt_autoencoder,
            flow_layers=flow_layers,
            flow_hidden_dim=flow_hidden_dim,
            hom_layers=hom_layers,
            hom_hidden_dim=hom_hidden_dim,
            num_bases_sym=num_bases_sym,
            num_bases_null=num_bases_null,
            scale_map=scale_map,
            eps_p_sym=eps_p_sym,
            eps_q_sym=eps_q_sym,
            eps_p_null=eps_p_null,
            eps_q_null=eps_q_null,
            normalize_generators=normalize_generators,
            normalize_precision=normalize_precision,
            rescale_eps=rescale_eps,
            fix_w_sym_to_commutative_rotation_basis=fix_w_sym_to_commutative_rotation_basis,
            predicted_factors=predicted_factors,
        )

        self.sample_num = sample_num
        if sample_image is None:
            sample_image = torch.zeros(1, 3, 64, 64)
        else:
            self.sample_image = torch.from_numpy(sample_image).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        self.generator_num = generator_num
        self.repr_dims = repr_dims

        self.lr = lr

        self.cov_cotangent = None
        self.cov_sym = None
        self.cov_null = None
        self.count = 0
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        try:
            output_dict = self.model.forward(batch)
        except CustomError as e:
            print(f"Skipping batch {batch_idx} due to error: {e}")
            return None
        log_prob_data = output_dict['log_prob_data'].mean()
        self.log('train_log_prob_data', log_prob_data, on_step=False, on_epoch=True, prog_bar=False)
        loss = -log_prob_data
        if 'log_prob_sym' in output_dict:
            log_prob_sym = output_dict['log_prob_sym'].mean()
            self.log('train_log_prob_sym', log_prob_sym, on_step=False, on_epoch=True, prog_bar=False)
            loss = loss - log_prob_sym
        if 'log_prob_null' in output_dict:
            log_prob_null = output_dict['log_prob_null'].mean()
            self.log('train_log_prob_null', log_prob_null, on_step=False, on_epoch=True, prog_bar=False)
            loss = loss - log_prob_null
        if 'kl_sym' in output_dict:
            kl_sym = output_dict['kl_sym'].mean()
            self.log('train_kl_sym', kl_sym, on_step=False, on_epoch=True, prog_bar=False)
            loss = loss + kl_sym
        if 'kl_null' in output_dict:
            kl_null = output_dict['kl_null'].mean()
            self.log('train_kl_null', kl_null, on_step=False, on_epoch=True, prog_bar=False)
            loss = loss + kl_null

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        try:
            output_dict = self.model.forward(batch)
        except CustomError as e:
            print(f"Skipping batch {batch_idx} due to error: {e}")
            return None
        log_prob_data = output_dict['log_prob_data'].mean()
        self.log('val_log_prob_data', log_prob_data, on_step=False, on_epoch=True, prog_bar=False)
        loss = -log_prob_data
        if 'log_prob_sym' in output_dict:
            log_prob_sym = output_dict['log_prob_sym'].mean()
            self.log('val_log_prob_sym', log_prob_sym, on_step=False, on_epoch=True, prog_bar=False)
            loss = loss - log_prob_sym
        if 'log_prob_null' in output_dict:
            log_prob_null = output_dict['log_prob_null'].mean()
            self.log('val_log_prob_null', log_prob_null, on_step=False, on_epoch=True, prog_bar=False)
            loss = loss - log_prob_null
        if 'kl_sym' in output_dict:
            kl_sym = output_dict['kl_sym'].mean()
            self.log('val_kl_sym', kl_sym, on_step=False, on_epoch=True, prog_bar=False)
            loss = loss + kl_sym
        if 'kl_null' in output_dict:
            kl_null = output_dict['kl_null'].mean()
            self.log('val_kl_null', kl_null, on_step=False, on_epoch=True, prog_bar=False)
            loss = loss + kl_null

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self._update_stats(
            output_dict.get('z'),
            output_dict.get('cv_sym'),
            output_dict.get('cv_null'),
        )
        return loss
    
    def _reset_stats(self):
        self.cov_cotangent = None
        self.cov_sym = None
        self.cov_null = None
        self.count = 0
    
    @torch.no_grad()
    def _update_stats(self, z: torch.Tensor, cv_sym: torch.Tensor, cv_null: torch.Tensor):
        batch_size = z.size(0)
        z = z.detach()
        cov_cotangent = torch.einsum('bi,bj->ij', cv_null, cv_null)
        if cv_sym is not None:
            cv_sym = cv_sym.detach()
            cov_sym = torch.einsum('bi,bj,bk,bl->ijkl', cv_sym, z, cv_sym, z)
        if cv_null is not None:
            cv_null = cv_null.detach()
            cov_null = torch.einsum('bi,bj,bk,bl->ijkl', cv_null, z, cv_null, z)

        if self.count == 0:
            self.cov_cotangent = cov_cotangent
            if cv_sym is not None:
                self.cov_sym = cov_sym
            if cv_null is not None:
                self.cov_null = cov_null
            self.count = batch_size
        else:
            self.cov_cotangent.add_(cov_cotangent)
            if cv_sym is not None:
                self.cov_sym.add_(cov_sym)
            if cv_null is not None:
                self.cov_null.add_(cov_null)
            self.count += batch_size

    @torch.no_grad()
    def _project_covariance(self):
        cov_cotangent = self.cov_cotangent / self.count
        l, v = torch.linalg.eigh(cov_cotangent)

        output_dict = dict(l=l, basis=v)

        if self.cov_sym is not None:
            cov_sym = self.cov_sym / self.count
            cov_sym_proj = torch.einsum('ijkl,ip,jq,kr,ls->pqrs', cov_sym, v, v, v, v)
            output_dict.update(cov_sym_proj=cov_sym_proj)
        if self.cov_null is not None:
            cov_null = self.cov_null / self.count
            cov_null_proj = torch.einsum('ijkl,ip,jq,kr,ls->pqrs', cov_null, v, v, v, v)
            output_dict.update(cov_null_proj=cov_null_proj)

        return output_dict
    
    @torch.no_grad()
    def _compute_generators(self, cov_proj: torch.Tensor, repr_basis: torch.Tensor, repr_dim: int):
        num_bases = repr_dim * (repr_dim - 1) // 2
        cov_proj = cov_proj[-repr_dim:, -repr_dim:, -repr_dim:, -repr_dim:]
        repr_basis = repr_basis[:, -repr_dim:]
        generator_basis = torch.zeros(num_bases, repr_dim, repr_dim, device=cov_proj.device)
        idx = 0
        for i in range(repr_dim):
            for j in range(i+1, repr_dim):
                generator_basis[idx, i, j] = 1.0 / math.sqrt(2)
                generator_basis[idx, j, i] = -1.0 / math.sqrt(2)
                idx += 1
        cov_proj = torch.einsum('ijkl,mij,nkl->mn', cov_proj, generator_basis, generator_basis)
        l, v = torch.linalg.eigh(cov_proj)
        generators = torch.einsum('nm,nij->mij', v, generator_basis)
        generators = torch.einsum('pi,qj,mij->mpq', repr_basis, repr_basis, generators)

        return l, generators
    
    @torch.no_grad()
    def _align_estimated_generators(self, W: torch.Tensor):
        L = W - W.mT  # (num_bases, input_dim, input_dim)
        u, s, vh = torch.linalg.svd(L.flatten(1, 2), full_matrices=False)
        L = vh.reshape(-1, L.size(1), L.size(2))
        return s, L
    
    @staticmethod
    def lie_algebra_loss(L: torch.Tensor) -> torch.Tensor:
        Z = torch.einsum('pik,qkj->pqij', L, L)
        Z = Z - Z.transpose(0, 1)
        C = torch.einsum('pqij,rij->prq', Z, L)
        R = Z - torch.einsum('prq,rij->pqij', C, L)
        res = R.square().sum((-2, -1))
        return res

    @staticmethod
    def lie_algebra_loss_curve(L: torch.Tensor) -> torch.Tensor:
        loss_list = []
        for i in range(1, L.size(0)+1):
            Li = L[:i]
            res = CLNFModule.lie_algebra_loss(Li)
            loss_list.append(res.max().item())
        return torch.tensor(loss_list, device=L.device)
    
    @torch.no_grad()
    def _transform(self, z: torch.Tensor, W: torch.Tensor):
        n = z.size(-1)
        h = W.size(0)

        w = W.view(h, 1, n, n)
        t = torch.linspace(-2, 2, 9).cuda().view(-1, 1, 1)
        w = torch.matrix_exp(t*w)
        w = w.view(-1, n, n)
        z = w @ z.unsqueeze(-1)
        z = z.squeeze(-1)

        x = self.model.decode(z)

        return x
    
    @torch.no_grad()
    def _sample(self):
        img = self.model.sample(self.sample_num)
        return img
    
    @torch.no_grad()
    def _plot_line(self, x: torch.Tensor, title="line_plot"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x.detach().cpu().numpy(), marker='o', ms=4)
        ax.set_title(title)
        self.logger.experiment.log({f"lines/{title}": wandb.Image(fig, caption=f"epoch {self.current_epoch}")})
        plt.close(fig)

    @torch.no_grad()
    def _plot_samples(self, img: torch.Tensor, nrow=8, title="sample"):
        img = img.clamp(0, 1)
        grid = tv.utils.make_grid(img.cpu(), nrow=nrow)
        self.logger.experiment.log({f"images/{title}": wandb.Image(grid, caption=f"epoch {self.current_epoch}")})

    @torch.no_grad()
    def _plot_generators(self, generators: torch.Tensor, title="symmetry_generators"):
        n = generators.size(0)
        ncol = min(8, n)
        nrow = math.ceil(n / ncol)

        fig, ax = plt.subplots(nrow, ncol, figsize=(4*ncol, 4*nrow))
        ax = ax.flatten()

        for i in range(generators.size(0)):
            im = ax[i].imshow(generators[i].detach().cpu().numpy(), cmap='bwr')
            ax[i].set_title(f'Generator {i+1}')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            fig.colorbar(im, ax=ax[i])

        self.logger.experiment.log({f"matrices/{title}": wandb.Image(fig, caption=f"epoch {self.current_epoch}")})
        plt.close(fig)
    
    def on_validation_epoch_end(self):
        if not hasattr(self.logger, "experiment"):
            return

        z = self.model.encode(self.sample_image)

        try:
            if self.model.W_sym is not None:
                s_sym, L_sym = self._align_estimated_generators(self.model.W_sym)
                curve_sym = self.lie_algebra_loss_curve(L_sym)
                self._plot_line(curve_sym, title="estimated/sym/lie_algebra_loss")
                L_sym = L_sym[:self.generator_num]
                self._plot_line(s_sym, title="estimated/sym/singular_values")
                x_sym = self._transform(z, L_sym)
                self._plot_samples(x_sym, nrow=9, title=f'estimated/sym')
                self._plot_generators(L_sym, title="estimated/sym")
            if self.model.W_null is not None:
                s_null, L_null = self._align_estimated_generators(self.model.W_null)
                curve_null = self.lie_algebra_loss_curve(L_null)
                self._plot_line(curve_null, title="estimated/null/lie_algebra_loss")
                L_null = L_null[:self.generator_num]
                self._plot_line(s_null, title="estimated/null/singular_values")
                x_null = self._transform(z, L_null)
                self._plot_samples(x_null, nrow=9, title=f'estimated/null')
                self._plot_generators(L_null, title="estimated/null")

        except Exception as e:
            print(f"Failed to compute estimated generators: {e}")

        if self.cov_cotangent is not None:
            try:
                output_dict = self._project_covariance()

                self._plot_line(output_dict['l'], title="analyzed/cotangent/eigenvalues")

                for repr_dim in self.repr_dims:
                    if 'cov_sym_proj' in output_dict:
                        l_sym, generators_sym = self._compute_generators(
                            output_dict['cov_sym_proj'], output_dict['basis'], repr_dim)
                        curve_sym = self.lie_algebra_loss_curve(generators_sym)
                        self._plot_line(curve_sym, title=f'analyzed/sym/repr_{repr_dim}/lie_algebra_loss')
                        generators_sym = generators_sym[:self.generator_num]
                        self._plot_line(l_sym, title=f'analyzed/sym/repr_{repr_dim}/eigenvalues')
                        x_sym = self._transform(z, generators_sym)
                        self._plot_samples(x_sym, nrow=9, title=f'analyzed/sym/repr_{repr_dim}')
                        self._plot_generators(generators_sym, title=f'analyzed/sym/repr_{repr_dim}')

                    if 'cov_null_proj' in output_dict:
                        l_null, generators_null = self._compute_generators(
                            output_dict['cov_null_proj'], output_dict['basis'], repr_dim)
                        curve_null = self.lie_algebra_loss_curve(generators_null)
                        self._plot_line(curve_null, title=f'analyzed/null/repr_{repr_dim}/lie_algebra_loss')
                        generators_null = generators_null[:self.generator_num]
                        self._plot_line(l_null, title=f'analyzed/null/repr_{repr_dim}/eigenvalues')
                        x_null = self._transform(z, generators_null)
                        self._plot_samples(x_null, nrow=9, title=f'analyzed/null/repr_{repr_dim}')
                        self._plot_generators(generators_null, title=f'analyzed/null/repr_{repr_dim}')

            except Exception as e:
                print(f"Failed to compute analyzed generators: {e}")

        self._reset_stats()

        self._plot_samples(self._sample(), nrow=8, title="sampled_images")
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=3e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.3)

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'interval': 'epoch',
            #     'frequency': 1,
            # }
        }
