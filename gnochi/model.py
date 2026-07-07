import os
import typing

import torch
import torch.nn as nn

from . import numpy_compat  # noqa: F401
from .pose_utils import default_device

try:
    import smplx
except ImportError:  # SMPL is optional for JSON-only inference.
    smplx = None

PathLike = typing.Union[str, bytes, os.PathLike]
LATENT_SIZE = 128


class PoseCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        latent = LATENT_SIZE

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(latent),
            nn.LeakyReLU(),
        )
        self.encoder_mean = nn.LazyLinear(latent)
        self.encoder_logvar = nn.LazyLinear(latent)

        self.decoder = nn.Sequential(
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
        )
        self.decoder_pose = nn.Sequential(
            nn.LazyLinear(23 * 6),
            nn.LeakyReLU(),
            nn.LazyLinear(23 * 6),
            nn.LeakyReLU(),
            nn.LazyLinear(23 * 6),
            nn.Unflatten(dim=-1, unflattened_size=(23, 6)),
        )
        self.decoder_trans = nn.Sequential(
            nn.LazyLinear(3),
            nn.LeakyReLU(),
            nn.LazyLinear(3),
            nn.LeakyReLU(),
            nn.LazyLinear(3),
        )
        self.decoder_rot = nn.Sequential(
            nn.LazyLinear(6),
            nn.LeakyReLU(),
            nn.LazyLinear(6),
            nn.LeakyReLU(),
            nn.LazyLinear(6),
        )

    def decode(
            self,
            encoded_pose: torch.Tensor,
            conditional_pose: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conditional_pose_flat = torch.flatten(conditional_pose, start_dim=1)
        concatenated = torch.cat([encoded_pose, conditional_pose_flat], dim=1)

        decoded_internal = self.decoder(concatenated)
        decoded_pose = self.decoder_pose(decoded_internal)
        decoded_trans = self.decoder_trans(decoded_internal)
        decoded_rot = self.decoder_rot(decoded_internal)
        return decoded_pose, decoded_trans, decoded_rot


class CapFix(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(23 * 6),
            nn.Unflatten(dim=-1, unflattened_size=(23, 6)),
        )

    def forward(
            self,
            input_pose_sixdof: torch.Tensor,
            input_global_orient_sixdof: torch.Tensor,
            input_transl: torch.Tensor,
            conditional_pose_sixdof: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                input_pose_sixdof.flatten(start_dim=-2),
                input_global_orient_sixdof,
                input_transl,
                conditional_pose_sixdof.flatten(start_dim=-2),
            ],
            dim=1,
        )
        return self.layers(x)


class GnocchiModel:
    def __init__(
            self,
            cvae_weights_path: PathLike,
            capfix_weights_path: typing.Optional[PathLike] = None,
            smpl_model_path: typing.Optional[PathLike] = None,
            device: typing.Optional[typing.Union[str, torch.device]] = None,
    ):
        self.device = torch.device(device) if device is not None else default_device()

        cvae = PoseCVAE().to(self.device)
        cvae_weights = torch.load(cvae_weights_path, map_location=self.device)
        cvae.load_state_dict(cvae_weights)
        cvae.eval()
        self.cvae = cvae

        self.capfix = None
        if capfix_weights_path:
            capfix = CapFix().to(self.device)
            capfix_weights = torch.load(capfix_weights_path, map_location=self.device)
            capfix.load_state_dict(capfix_weights)
            capfix.eval()
            self.capfix = capfix

        self.smpl = None
        if smpl_model_path:
            if smplx is None:
                raise ImportError("smplx is required for mesh export. Install requirements.txt first.")
            self.smpl = smplx.build_layer(smpl_model_path).to(self.device)
            self.smpl.eval()

    @property
    def has_capfix(self) -> bool:
        return self.capfix is not None

    @property
    def has_smpl(self) -> bool:
        return self.smpl is not None
