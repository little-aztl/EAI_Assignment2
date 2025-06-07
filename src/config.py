from dataclasses import dataclass, field  # type: ignore
from typing import Dict, List, Any
import numpy as np
import yaml

# You can modify Config according to your need
# if you add some non-hashable types like list or dict, you need to use field
# like xxx: list = field(default_factory=lambda: [1, 2, 3])

DEFAULT_MODEL_HYPERPARAMS = {
    "est_pose": {
        "encoder_hidden_dims": [64, 128, 256],
        'activation': 'ReLU',
        'decoder_hidden_dims': [128],
        'loss': {
            'rotation_weight': 0.5,
            'translation_weight': 0.5,
        }
    },
    "est_coord": {
        "light_encoder_hidden_dim": 64,
        "encoder_hidden_dims": [128, 1024],
        'activation': 'ReLU',
        'decoder_hidden_dims': [512, 256],
        "RANSAC": {
            'inlier_thresh': 0.010,
            "max_iter": 1000,
            "n_samples": 4,
        }
    }
}

@dataclass
class Config:
    model_type: str = None
    """can be est_pose or est_coord"""
    exp_name: str = "debug"
    """if exp_name is debug, it won't be logged in wandb"""
    robot: str = "galbot"
    """the robot we are using"""
    obj_name: str = "power_drill"
    """the object we want to grasp"""
    checkpoint: str = None
    """if not None, then we will continue training from this checkpoint"""
    max_iter: int = 10000
    """the maximum number of iterations"""
    batch_size: int = 16
    """the batch size for training"""
    learning_rate: float = 1e-3
    """maximum (and initial) learning rates"""
    learning_rate_min: float = 1e-8
    """we use cosine decay for learning rate, and this is the minimum (and final) learning rate"""
    log_interval: int = 100
    """log for each log_interval iterations"""
    save_interval: int = 2500
    """save the model every save_interval iterations"""
    val_interval: int = 500
    """get metric from validation set every val_interval iterations"""
    val_num: int = 10
    """run val_num batches for each validation to get stable result"""
    num_workers: int = 8
    """how many workers to use for data loading, if you are debugging, use 0 so that it won't create new processes"""
    seed: int = 0
    """the random seed for training"""
    device: str = "cpu"
    """the device to use for training, you can use cuda:0 if you have a gpu"""
    point_num: int = 1024
    """number of points sampled from the full observation point cloud"""

    model_hyperparams: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.model_hyperparams:
            self.model_hyperparams = DEFAULT_MODEL_HYPERPARAMS.get(self.model_type, {})

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from a YAML file.

        Use default values for any missing keys in the YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML file.

        Returns
        -------
        Config
            An instance of the Config class with values loaded from the YAML file.
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)
