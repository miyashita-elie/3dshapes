from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import pytorch_lightning as pl

from differentiable_3dshapes import Differentiable3Dshapes


FactorBatch = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class FactorRangeConfig:
    size_min: float = 0.75
    size_max: float = 1.25


def _sample_factors(n: int, *, generator: torch.Generator, factor_range: FactorRangeConfig) -> FactorBatch:
    shape = torch.randint(0, 4, (n,), generator=generator, dtype=torch.int64).to(torch.float32)
    size = torch.empty(n, dtype=torch.float32).uniform_(factor_range.size_min, factor_range.size_max, generator=generator)
    orientation = torch.empty(n, dtype=torch.float32).uniform_(0.0, 1.0, generator=generator)
    floor_hue = torch.empty(n, dtype=torch.float32).uniform_(0.0, 1.0, generator=generator)
    wall_hue = torch.empty(n, dtype=torch.float32).uniform_(0.0, 1.0, generator=generator)
    object_hue = torch.empty(n, dtype=torch.float32).uniform_(0.0, 1.0, generator=generator)
    return {
        "shape": shape,
        "size": size,
        "orientation": orientation,
        "floor_hue": floor_hue,
        "wall_hue": wall_hue,
        "object_hue": object_hue,
    }


class FactorSamplingDataset(torch.utils.data.Dataset):
    """Dataset of factors only. Can be fixed or dynamically sampled on each access."""

    def __init__(
        self,
        *,
        length: int,
        factor_range: FactorRangeConfig,
        seed: int,
        dynamic: bool,
    ) -> None:
        self.length = int(length)
        self.factor_range = factor_range
        self.seed = int(seed)
        self.dynamic = bool(dynamic)

        self._fixed: Optional[FactorBatch]
        self._generator: Optional[torch.Generator]
        if self.dynamic:
            self._fixed = None
            self._generator = torch.Generator().manual_seed(self.seed)
        else:
            g = torch.Generator().manual_seed(self.seed)
            self._fixed = _sample_factors(self.length, generator=g, factor_range=self.factor_range)
            self._generator = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> FactorBatch:
        if self._fixed is not None:
            return {k: v[idx] for k, v in self._fixed.items()}

        if self._generator is None:
            raise RuntimeError("Dynamic dataset generator is not initialized.")
        # Dynamic mode ignores idx and draws IID factors from one persistent RNG stream.
        sample = _sample_factors(1, generator=self._generator, factor_range=self.factor_range)
        return {k: v[0] for k, v in sample.items()}


class _RendererCollate:
    def __init__(
        self,
        *,
        return_grad: bool,
        device: torch.device,
    ) -> None:
        self.return_grad = return_grad
        self.device = device
        self.renderer = Differentiable3Dshapes().to(self.device)
        self.renderer.eval()

    def __call__(self, batch: Iterable[FactorBatch]) -> dict:
        items = list(batch)
        factors = {
            "shape": torch.stack([x["shape"] for x in items], dim=0),
            "size": torch.stack([x["size"] for x in items], dim=0),
            "orientation": torch.stack([x["orientation"] for x in items], dim=0),
            "floor_hue": torch.stack([x["floor_hue"] for x in items], dim=0),
            "wall_hue": torch.stack([x["wall_hue"] for x in items], dim=0),
            "object_hue": torch.stack([x["object_hue"] for x in items], dim=0),
        }
        factors = {k: v.to(self.device, non_blocking=True) for k, v in factors.items()}

        if self.return_grad:
            jacobians, image = self.renderer(
                shape=factors["shape"],
                size=factors["size"],
                orientation=factors["orientation"],
                floor_hue=factors["floor_hue"],
                wall_hue=factors["wall_hue"],
                object_hue=factors["object_hue"],
                return_grad=True,
            )
        else:
            image = self.renderer(
                shape=factors["shape"],
                size=factors["size"],
                orientation=factors["orientation"],
                floor_hue=factors["floor_hue"],
                wall_hue=factors["wall_hue"],
                object_hue=factors["object_hue"],
                return_grad=False,
            )
            jacobians = None

        out = {"image": image, "factors": factors}
        if jacobians is not None:
            out["jacobians"] = jacobians
        return out


class Differentiable3DShapesDataModule(pl.LightningDataModule):
    """Factor-sampling DataModule with batched rendering after collation."""

    def __init__(
        self,
        *,
        train_samples_per_epoch: int = 480000,
        val_samples: int = 24000,
        batch_size: int = 256,
        num_workers: int = 0,
        train_resample_each_epoch: bool = True,
        return_grad: bool = False,
        seed: int = 0,
        render_device: Optional[str] = None,
        factor_range: FactorRangeConfig = FactorRangeConfig(),
    ) -> None:
        super().__init__()
        if num_workers != 0:
            raise ValueError("GPU rendering is done in collate_fn, so set num_workers=0.")

        self.train_samples_per_epoch = int(train_samples_per_epoch)
        self.val_samples = int(val_samples)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.train_resample_each_epoch = bool(train_resample_each_epoch)
        self.return_grad = bool(return_grad)
        self.seed = int(seed)
        self.factor_range = factor_range

        if render_device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required by default. Pass render_device='cpu' to run on CPU.")
            render_device = "cuda"
        self.render_device = torch.device(render_device)

        self._train_ds: Optional[FactorSamplingDataset] = None
        self._val_ds: Optional[FactorSamplingDataset] = None
        self._train_collate: Optional[_RendererCollate] = None
        self._val_collate: Optional[_RendererCollate] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_ds is None:
            self._train_ds = FactorSamplingDataset(
                length=self.train_samples_per_epoch,
                factor_range=self.factor_range,
                seed=self.seed,
                dynamic=self.train_resample_each_epoch,
            )

        if self._val_ds is None:
            self._val_ds = FactorSamplingDataset(
                length=self.val_samples,
                factor_range=self.factor_range,
                seed=self.seed + 1,
                dynamic=False,
            )

        if self._train_collate is None:
            self._train_collate = _RendererCollate(
                return_grad=self.return_grad,
                device=self.render_device,
            )
        if self._val_collate is None:
            self._val_collate = _RendererCollate(
                return_grad=self.return_grad,
                device=self.render_device,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self._train_ds is None or self._train_collate is None:
            self.setup(stage="fit")
        assert self._train_ds is not None
        assert self._train_collate is not None
        return torch.utils.data.DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._train_collate,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self._val_ds is None or self._val_collate is None:
            self.setup(stage="fit")
        assert self._val_ds is not None
        assert self._val_collate is not None
        return torch.utils.data.DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._val_collate,
            pin_memory=False,
            drop_last=False,
        )
