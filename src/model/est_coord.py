from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

from ..config import Config
from ..vis import Vis


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config

        self.shallow_encoder = nn.Linear(
            3, config.model_hyperparams['light_encoder_hidden_dim']
        )

        encoder_dims = [
            config.model_hyperparams['light_encoder_hidden_dim'],
            *config.model_hyperparams['encoder_hidden_dims']
        ]
        self.encoder = []
        for in_dim, out_dim in zip(encoder_dims[:-1], encoder_dims[1:]):
            self.encoder.append(getattr(nn, config.model_hyperparams['activation'])())
            self.encoder.append(nn.Linear(in_dim, out_dim))
        self.encoder = nn.Sequential(*self.encoder)


        decoder_dims = [
            config.model_hyperparams['light_encoder_hidden_dim'] + encoder_dims[-1],
            *config.model_hyperparams['decoder_hidden_dims'],
            3
        ]
        self.decoder = []
        for mlp_index, (in_dim, out_dim) in enumerate(zip(decoder_dims[:-1], decoder_dims[1:])):
            self.decoder.append(nn.Linear(in_dim, out_dim))
            if mlp_index < len(decoder_dims) - 2:
                self.decoder.append(getattr(nn, config.model_hyperparams['activation'])())
        self.decoder = nn.Sequential(*self.decoder)

        self.distance_loss = nn.MSELoss()

        self.helper_tensor = torch.tensor([1.0, 0, 0, 0, 1.0, 0, 0, 0])


    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        batch_size, num_points = pc.shape[0], pc.shape[1]

        shallow_point_feature = self.shallow_encoder(pc) # (B, N, shallow_D)
        heavy_point_feature = self.encoder(shallow_point_feature) # (B, N, D)
        global_feature = torch.max(heavy_point_feature, dim=1)[0] # (B, D)

        point_feature = torch.cat([
            shallow_point_feature, # (B, N, shallow_D)
            global_feature[:, torch.newaxis, :].expand(-1, num_points, -1), # (B, N, D)
        ], dim=2) # (B, N, D + shallow_D)

        pred_coord = self.decoder(point_feature) # (B, N, 3)


        loss = self.distance_loss(pred_coord, coord)
        distance_error = torch.linalg.norm(pred_coord - coord, dim=2).mean()
        metric = dict(
            loss=loss,
            # additional metrics you want to log
            trans_error=distance_error,
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

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        batch_size, num_points = pc.shape[0], pc.shape[1]

        shallow_point_feature = self.shallow_encoder(pc) # (B, N, shallow_D)
        heavy_point_feature = self.encoder(shallow_point_feature) # (B, N, D)
        global_feature = torch.max(heavy_point_feature, dim=1)[0] # (B, D)

        point_feature = torch.cat([
            shallow_point_feature, # (B, N, shallow_D)
            global_feature[:, torch.newaxis, :].expand(-1, num_points, -1), # (B, N, D)
        ], dim=2) # (B, N, D + shallow_D)

        pred_coord = self.decoder(point_feature) # (B, N, 3)

        center_in_camera = torch.mean(pc, dim=1) # (B, 3)
        center_in_object = torch.mean(pred_coord, dim=1) # (B, 3)
        pred_trans = center_in_camera - center_in_object # (B, 3)

        pc_in_camera = pc - center_in_camera[:, torch.newaxis, :] # (B, N, 3)
        pc_in_object = pred_coord - center_in_object[:, torch.newaxis, :] # (B, N, 3)

        tmp = torch.matmul(
            pc_in_camera.transpose(1, 2), # (B, 3, N)
            pc_in_object # (B, N, 3)
        ) # (B, 3, 3)
        Q, S, Vh = torch.linalg.svd(tmp) # Q: (B, 3, 3), S: (B, 3), Vh: (B, 3, 3)
        det_u, det_vh = torch.linalg.det(Q), torch.linalg.det(Vh) # det_u: (B,), det_vh: (B,)
        pred_rotation = torch.matmul(
            Q,
            torch.cat([
                self.helper_tensor[torch.newaxis].to(pc.device).expand(batch_size, -1), # (B, 8)
                (det_u * det_vh)[:, torch.newaxis] # (B, 1)
            ], dim=1).reshape(batch_size, 3, 3) # (B, 3, 3)
        ) # (B, 3, 3)
        pred_rotation = torch.matmul(pred_rotation, Vh) # (B, 3, 3)

        return pred_trans, pred_rotation