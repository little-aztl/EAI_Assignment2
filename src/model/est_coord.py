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

        self.ransac_config = config.model_hyperparams['RANSAC']


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

        def fitting(goal_pc:torch.Tensor, cur_pc:torch.Tensor):
            '''
            goal_pc: (B, n_samples, 3)
            cur_pc: (B, n_samples, 3)
            '''
            goal_center = goal_pc.mean(dim=1) # (B, 3)
            cur_center = cur_pc.mean(dim=1) # (B, 3)


            normalized_goal_pc = goal_pc - goal_center[:, torch.newaxis, :] # (B, n_samples, 3)
            normalized_cur_pc = cur_pc - cur_center[:, torch.newaxis, :] # (B, n_samples, 3)

            tmp = torch.matmul(
                normalized_goal_pc.transpose(1, 2), # (B, 3, n_samples)
                normalized_cur_pc # (B, n_samples, 3)
            ) # (B, 3, 3)
            U, _, Vh = torch.linalg.svd(tmp) # U: (B, 3, 3), Vh: (B, 3, 3)
            det_u, det_v = torch.linalg.det(U), torch.linalg.det(Vh) # det_u: (B,) det_v: (B,)
            det = det_u * det_v # (B,)
            Vh[:, 2, :] = Vh[:, 2, :] * det[:, torch.newaxis]

            pred_rotation = torch.matmul(U, Vh) # (B, 3, 3)
            pred_translation = goal_center - torch.matmul(pred_rotation, cur_center[:, :, torch.newaxis]).squeeze(dim=2) # (B, 3)

            return pred_rotation, pred_translation

        n_samples = self.ransac_config['n_samples']
        max_iter = self.ransac_config['max_iter']

        selected_indices = torch.randint(
            low=0, high=num_points,
            size=(batch_size, max_iter, n_samples),
            device=pc.device
        ) # (B, max_iter, n_samples)

        selected_goal_pc = torch.gather(
            input=pc[:, torch.newaxis].expand(-1, max_iter, -1, -1), # (B, max_iter, N, 3)
            dim=2,
            index=selected_indices[:, :, :, torch.newaxis].expand(-1, -1, -1, 3) # (B, max_iter, n_samples, 3)
        ).reshape(-1, n_samples, 3) # (B * max_iter, n_samples, 3)
        selected_cur_pc = torch.gather(
            input=pred_coord[:, torch.newaxis].expand(-1, max_iter, -1, -1), # (B, max_iter, N, 3)
            dim=2,
            index=selected_indices[:, :, :, torch.newaxis].expand(-1, -1, -1, 3) # (B, max_iter, n_samples, 3)
        ).reshape(-1, n_samples, 3) # (B * max_iter, n_samples, 3)

        pred_rotations, pred_translations = fitting(selected_goal_pc, selected_cur_pc) # pred_rotation: (B * max_iter, 3, 3), pred_translation: (B * max_iter, 3)

        pred_rotations = pred_rotations.reshape(batch_size, max_iter, 3, 3) # (B, max_iter, 3, 3)
        pred_translations = pred_translations.reshape(batch_size, max_iter, 3) # (B, max_iter, 3)

        transformed_cur_pc = torch.matmul(
            pred_rotations[:, :, torch.newaxis, :, :], # (B, max_iter, 1, 3, 3)
            pred_coord[:, torch.newaxis, :, :, torch.newaxis] # (B, 1, N, 3, 1)
        ).squeeze(dim=-1) + pred_translations[:, :, torch.newaxis, :] # (B, max_iter, N, 3)
        distance_error = torch.linalg.norm(
            pc[:, torch.newaxis, :, :] - transformed_cur_pc, # (B, max_iter, N, 3)
            dim=3
        ) # (B, max_iter, N)
        is_inlier = distance_error < self.ransac_config['inlier_thresh'] # (B, max_iter, N)
        inliers_count = is_inlier.sum(dim=2) # (B, max_iter)
        best_iter = inliers_count.argmax(dim=1) # (B,)

        is_inlier = torch.gather(
            input=is_inlier, # (B, max_iter, N)
            dim=1,
            index=best_iter[:, torch.newaxis, torch.newaxis].expand(-1, -1, num_points) # (B, 1, N)
        ).squeeze(dim=1) # (B, N)

        ret_rotation, ret_translation = [], []
        for b in range(batch_size):
            selected_goal_pc = pc[b, is_inlier[b]] # (n_inliers, 3)
            selected_cur_pc = pred_coord[b, is_inlier[b]] # (n_inliers, 3)


            pred_rotation, pred_translation = fitting(selected_goal_pc[torch.newaxis], selected_cur_pc[torch.newaxis]) # pred_rotation: (3, 3), pred_translation: (3,)
            ret_rotation.append(pred_rotation.squeeze(dim=0))
            ret_translation.append(pred_translation.squeeze(dim=0))

        ret_rotation = torch.stack(ret_rotation, dim=0) # (B, 3, 3)
        ret_translation = torch.stack(ret_translation, dim=0) # (B, 3)
        return ret_translation, ret_rotation

