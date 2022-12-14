"""
Sample configuration dataclasses for training repo.

The python built-in `dataclass` decorator adds several features to a class, including the ability
to automatically generate and specialize __init__ and __repr__, and the ability to easily access 
the class's fields using the `.` operator.

Here we also use dataclasses to separate groups of configuration parameters from one another and
enable hierarchial relationships.

The comments within the configuration dataclasses are able to be read, tokenized, and used to
form the help documentation for a CLI.
"""
from typing import List, Optional
from dataclasses import dataclass, field
import numpy as np
import pyrallis


@dataclass()
class DataConfig:
    """Configuration for dataset specification"""

    # Path to a directory holding all datasets.
    dataset_root: str = "/home/matthew.phelps/data/datasets/dasie/jfletcher"
    # Path to the train data TFRecords directory.
    dataset_name: str = "speedplus"


@dataclass()
class LogConfig:
    """Configuration for logging (mlflow) specification"""

    # Mlflow tracking uri
    uri: Optional[str] = "~/dev/DASIE/jfletcher/dasie/mlruns"
    # MLflow run name.
    run_name: str = "run_0"
    # MLFlow experiment name.
    exp_name: str = "debug"
    # Toggle asynchronous logging (not compatible with ray tune)
    enable_async: bool = True
    # Number of threads to use in async logging (2 threads/core typically)
    num_threads: int = 4
    # Every `train_freq` steps, log training quantities (metrics, single image batch, etc.)
    train_freq: int = 1
    # Run evaluation before first train step
    evaluate_init: bool = True
    # Every `save_freq` epochs save the model checkpoint
    save_freq: int = 1
    # Save initial model state
    save_init: bool = False
    # Save last model state
    save_last: bool = False
    # Save best model state (early stopping)
    save_best: bool = False
    # Log images
    images: bool = True
    # Show the plot?
    show_plot: bool = False
    # Save the plot? (jf: true)
    save_plot: bool = True
    # Number of epochs to wait before plotting (jf: 64)
    plot_freq: int = 1


@dataclass()
class TrainConfig:
    """Configuration for training instance"""

    # Configuration for logging specification
    log: LogConfig = field(default_factory=LogConfig)
    # Configuration for dataset specification
    data: DataConfig = field(default_factory=DataConfig)
    # GPU list to expose to training instance.
    gpu_list: List[int] = field(default_factory=lambda: [0])
    # The loss function used.
    loss_name: str = "mae"
    # The number of optimization steps to perform. (jf: 4096)
    num_steps: int = 2
    # A random seed for repeatability.
    random_seed: int = np.random.randint(0, 2048)
    # Number of DASIE subapertures.
    num_subapertures: int = 15
    # Meters of space between subapertures.
    subaperture_spacing_meters: float = 0.1
    # Number of Zernike terms to simulate.
    num_zernike_indices: int = 1
    # Diameter of DA and mono apertures in meters.
    aperture_diameter_meters: float = 2.5
    # The size of the optimizer step.
    learning_rate: float = 0.0001
    # Quantiziation of all images. (jf: 256)
    spatial_quantization: int = 128
    # Number of perfect images per batch. (jf: 4)
    batch_size: int = 1
    # The number of sequential frames to model.
    num_exposures: int = 1
    # If true, crop images to spatial quantization.
    crop: bool = True
    # If true, use MTF, image spectrum product, else use PSF convolution.
    hadamard_image_formation: bool = False
    # The angular scale/pixel of the object plane.
    object_plane_scale: float = 1.0
    # Base filter size for recovery model. (jf: 16)
    recovery_model_filter_scale: int = 2
    # If true, each subaperture is constrained such that only the Zernike coefficient with the 
    # same index as the subaperture index is none-zero.
    zernike_debug: bool = False


if __name__ == "__main__":
    """Test the train config, export defaults to resources/train_cfg.yaml"""
    cfg = pyrallis.parse(config_class=TrainConfig)
    pyrallis.dump(cfg, open("../resources/train_cfg.yaml", "w"))