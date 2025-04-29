from typing import Tuple, Dict
import torch
from torch import nn

from ..config import Config
from ..utils import safe_acos


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config
        encoder_dims = [3, *config.model_hyperparams['encoder_hidden_dims']]
        self.encoder = []
        for mlp_index, (in_dim, out_dim) in enumerate(zip(encoder_dims[:-1], encoder_dims[1:])):
            self.encoder.append(nn.Linear(in_dim, out_dim))
            if mlp_index < len(encoder_dims) - 2:
                self.encoder.append(getattr(nn, config.model_hyperparams['activation'])())
        self.encoder = nn.Sequential(*self.encoder)

        decoder_dims = [encoder_dims[-1], *config.model_hyperparams['decoder_hidden_dims'], 12]
        self.decoder = []
        for mlp_index, (in_dim, out_dim) in enumerate(zip(decoder_dims[:-1], decoder_dims[1:])):
            self.decoder.append(nn.Linear(in_dim, out_dim))
            if mlp_index < len(decoder_dims) - 2:
                self.decoder.append(getattr(nn, config.model_hyperparams['activation'])())
        self.decoder = nn.Sequential(*self.decoder)

        self.transloss = nn.MSELoss()
        self.transloss_weight = config.model_hyperparams['loss']['translation_weight']
        self.rotloss = nn.MSELoss()
        self.rotloss_weight = config.model_hyperparams['loss']['rotation_weight']

        self.helper_tensor = torch.tensor([1.0, 0, 0, 0, 1.0, 0, 0, 0])


    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        # raise NotImplementedError("You need to implement the forward function")
        batch_size = pc.shape[0]
        device = pc.device

        point_embedding = self.encoder(pc) # (B, N, C)
        global_embedding = torch.max(point_embedding, dim=1)[0] # (B, C)
        prediction = self.decoder(global_embedding) # (B, 12)

        rotation = prediction[:, :9].reshape(-1, 3, 3) # (B, 3, 3)
        translation = prediction[:, 9:] # (B, 3)

        transloss = self.transloss(translation, trans)
        rotloss = self.rotloss(rotation, rot)
        loss = self.transloss_weight * transloss + self.rotloss_weight * rotloss
        trans_error = torch.norm(translation - trans, dim=1).mean()

        U, _, Vh = torch.linalg.svd(rotation)
        det_u, det_vh = torch.linalg.det(U), torch.linalg.det(Vh)
        tmp = (det_u * det_vh)[:, torch.newaxis] # (B, 1)
        tmp = torch.concat([
            torch.tile(self.helper_tensor[torch.newaxis, :].to(device), (batch_size, 1)), # (B, 8)
            tmp # (B, 1)
        ], dim=1).reshape(batch_size, 3, 3) # (B, 3, 3)
        rotation = torch.matmul(U, tmp) # (B, 3, 3)
        rotation = torch.matmul(rotation, Vh) # (B, 3, 3)

        rot_error = torch.sum(torch.diagonal(torch.matmul(rotation, torch.transpose(rot, 1, 2)), dim1=-2, dim2=-1), dim=-1) # (B,)
        rot_error = (rot_error - 1) / 2 # (B,)
        rot_error = safe_acos(rot_error).mean()

        metric = dict(
            loss=loss,
            trans_loss=transloss,
            rot_loss=rotloss,
            # additional metrics you want to log
            trans_error=trans_error,
            rot_error=rot_error,
        )
        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.
        """
        batch_size = pc.shape[0]
        device = pc.device

        point_embedding = self.encoder(pc) # (B, N, C)
        global_embedding = torch.max(point_embedding, dim=1)[0] # (B, C)
        prediction = self.decoder(global_embedding) # (B, 12)

        rotation = prediction[:, :9].reshape(-1, 3, 3) # (B, 3, 3)
        translation = prediction[:, 9:] # (B, 3)

        U, _, Vh = torch.linalg.svd(rotation)
        det_u, det_vh = torch.linalg.det(U), torch.linalg.det(Vh)
        tmp = (det_u * det_vh)[:, torch.newaxis] # (B, 1)
        tmp = torch.concat([
            torch.tile(self.helper_tensor[torch.newaxis, :].to(device), (batch_size, 1)), # (B, 8)
            tmp # (B, 1)
        ], dim=1).reshape(batch_size, 3, 3) # (B, 3, 3)
        rotation = torch.matmul(U, tmp) # (B, 3, 3)
        rotation = torch.matmul(rotation, Vh) # (B, 3, 3)

        return translation, rotation
