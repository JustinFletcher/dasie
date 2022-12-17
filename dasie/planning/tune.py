"""
Uses ray tune to perform hyparameter search, optimization, and run scheduling

Note: ray.tune cannot serialize dataclasses with enum fields; thus any object depending on 
    cfg must be constructed entirely within the primary run function

Usage:
Single gpu:
    CUDA_VISIBLE_DEVICES=0 tune.py --log.exp_id hyperparam_experiment_1
Multi gpu:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 tune.py --log.exp_id hyperparam_experiment_1
Under `tune_function`, specify the number of GPUs and CPU's to use *per* training run.

Hyperparms you wish to vary are defined in the `tune_config` and can be instantiated from this list
of objects: https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs .

To do a hyperparmater search (HyperOpt algorithim), simply uncomment `search_alg` under tune.run.

See the `tune` folder within the MlFlow UI for stderr, stdout logs and checkpoint files.
"""
import logging
from pathlib import Path

import mlflow
import numpy as np
import pyrallis
import ray
import tensorflow as tf
from cfg import TrainConfig
from differentiable_dasie import DASIEModel
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.suggest.hyperopt import HyperOptSearch
from recorder import Recorder
from train import generate_datasets, train


def run(tune_config, checkpoint_dir=None, cfg=None):
    """Execute training, testing, or inference run based on cfg"""
    # Set up logging (within an individual run)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="[%(asctime)s] (%(levelname)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Instantiate recorder and set to current experiment/run
    recorder = Recorder(cfg)
    recorder.set_experiment()
    recorder.set_run()

    ## Derive optical parameters
    # Compute the scaling factor from meters to alpha for a GG PDF over meters.
    ap_radius_meters = cfg.aperture_diameter_meters / 2
    subap_radius_meters = (
        (ap_radius_meters * np.sin(np.pi / cfg.num_subapertures))
        - (cfg.subaperture_spacing_meters / 2)
    ) / (1 + np.sin(np.pi / cfg.num_subapertures))
    subap_area = np.pi * (subap_radius_meters**2)
    total_subap_area = subap_area * cfg.num_subapertures
    mono_ap_area = np.pi * (ap_radius_meters**2)
    mono_to_dist_aperture_ratio = mono_ap_area / total_subap_area
    derived_optical_params = {
        "mono_to_dist_aperture_ratio": mono_to_dist_aperture_ratio,
        "ap_radius_meters": ap_radius_meters,
        "subap_radius_meters": subap_radius_meters,
    }

    # Set the crop size to the spatial quantization scale.
    if cfg.crop:
        crop_size = cfg.spatial_quantization
    else:
        crop_size = None

    # Log source files to mlflow
    recorder.log_files()
    # Log all TrainConfig parameters for this run to mlflow
    recorder.log_run_params()
    # Also log some derived params
    recorder.log_params(derived_optical_params)
    # Set directory for exporting artifacts (tfeventsfiles, images, etc)
    save_dir = str(recorder.root)

    # `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint should be
    # restored
    if checkpoint_dir:
        # Insert logic here to load a model given `checkpoint_dir`
        # This is only necessary if using a Ray Scheduler (stop/resumes runs based on performance)
        # Since this resumes a run, the optimizer gradients would also need to be restored
        pass

    # Begin by creating a new session.
    with tf.compat.v1.Session() as sess:

        logger.info("\n\n\n\n\n\n\n\n\n Session Created \n\n\n\n\n\n\n\n\n")

        # Set all our seeds.
        np.random.seed(cfg.random_seed)
        tf.compat.v1.set_random_seed(cfg.random_seed)

        # Make summary management variables.
        step = tf.Variable(0, dtype=tf.int64)
        step_update = step.assign_add(1)
        tf.summary.experimental.set_step(step)
        writer = tf.summary.create_file_writer(save_dir)

        # Create datasets
        logger.info("\n\n\n\n\n\n\n\n\n Loading datasets \n\n\n\n\n\n\n\n\n")
        train_dataset, valid_dataset = generate_datasets(
            cfg.data.dataset_root, cfg.data.dataset_name, cfg.batch_size, crop_size
        )

        # Get the image shapes stored during dataset construction.
        image_x_scale = train_dataset.image_shape[0]
        image_y_scale = train_dataset.image_shape[1]

        # Build a DA model.
        dasie_model = DASIEModel(
            sess,
            writer=writer,
            batch_size=cfg.batch_size,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            num_exposures=cfg.num_exposures,
            spatial_quantization=cfg.spatial_quantization,
            image_x_scale=image_x_scale,
            image_y_scale=image_y_scale,
            learning_rate=cfg.learning_rate,
            diameter_meters=cfg.aperture_diameter_meters,
            num_apertures=cfg.num_subapertures,
            subaperture_radius_meters=subap_radius_meters,
            recovery_model_filter_scale=cfg.recovery_model_filter_scale,
            loss_name=cfg.loss_name,
            num_zernike_indices=cfg.num_zernike_indices,
            hadamard_image_formation=cfg.hadamard_image_formation,
            zernike_debug=cfg.zernike_debug,
            recorder=recorder,
        )

        # Merge all the summaries from the graphs, flush and init the nodes.
        all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
        writer_flush = writer.flush()
        sess.run([writer.init(), step.initializer])

        # Optimize the DASIE model parameters.
        train(
            sess,
            dasie_model,
            train_dataset,
            valid_dataset,
            num_steps=cfg.num_steps,
            plot_periodicity=cfg.log.plot_freq,
            writer=writer,
            step_update=step_update,
            all_summary_ops=all_summary_ops,
            writer_flush=writer_flush,
            logdir=save_dir,
            save_plot=cfg.log.save_plot,
            show_plot=cfg.log.show_plot,
            # results_dict=base_results_dict,
            results_dict=None,
            recorder=recorder,
            ray_tune=True,
        )

    # Stop mlflow run, exit gracefully
    recorder.end_run()


def tune_function(cfg):
    """Defines run_fn, sets scheduler, and executes tune.run"""
    # Rather than wrapping a partial, simply wrap defined run_fn here
    # This sets tracking uri according to config
    @mlflow_mixin
    def run_fn(tune_config, checkpoint_dir=None):
        return run(tune_config, checkpoint_dir, cfg)

    # Default trial names are obscene, lets tame it down
    def trial_dir_str(trial):
        return "{}_{}".format(trial.trainable_name, trial.trial_id)

    # This reporter omits extraneous mlflow information in progress report
    # Also allows tables to persist at low verbose levels
    reporter = CLIReporter(
        parameter_columns=["batch_size", "lr"],
        print_intermediate_tables=True,
    )

    # Define hyperparmeter search algorithm
    hyperopt_search = HyperOptSearch(metric="valid_loss", mode="min")

    # Define the mlflow config and hyperparameter sampling
    tune_config = {
        "mlflow": {
            "tracking_uri": str(cfg.log.uri),
            "experiment_name": cfg.log.exp_name,
        },
        "lr": tune.loguniform(1e-7, 1e-2),
        "batch_size": tune.choice([1, 2, 3]),
    }

    # Execute parallel hyperparam tuning
    tune.run(
        run_fn,
        name="tuneup",
        num_samples=3,
        config=tune_config,
        resources_per_trial={"cpu": 2, "gpu": 1},
        # search_alg=hyperopt_search,
        keep_checkpoints_num=1,  # 2
        progress_reporter=reporter,
        verbose=1,
        local_dir="../../ray_results",  # parent dir must match mlflow uri parent dir!
        log_to_file=("stdout.log", "stderr.log"),
        trial_dirname_creator=trial_dir_str,
    )


def setup_experiment(cfg):
    """Setup ray tune, cfg, and mlflow for hyperparam experiment"""
    # Disable internal log and print statements convoluting ray cli logs
    ray.init(log_to_driver=False)

    # Force async to false as this breaks ray tune; force tune
    cfg.log.enable_async = False

    # Make cfg paths absolute
    cfg.data.dataset_root = Path(cfg.data.dataset_root).expanduser().absolute()
    cfg.log.uri = Path(cfg.log.uri).expanduser().absolute()

    # Set mlflow tracking URI
    uri = "file://" + str(cfg.log.uri)
    print(f"Setting tracking URI: {uri}")
    mlflow.set_tracking_uri(uri)

    # Create experiment if it does not exist
    if cfg.log.exp_name is not None:
        if mlflow.get_experiment_by_name(cfg.log.exp_name) is None:
            print(f"creating mlflow experiment: {cfg.log.exp_name}")
            mlflow.create_experiment(cfg.log.exp_name)

    return cfg


def main():
    # Parse config from CLI
    cfg = pyrallis.parse(config_class=TrainConfig)

    # Prepare cfg for mlflow hyperparam experiment
    cfg = setup_experiment(cfg)

    # Run Ray Tune
    tune_function(cfg)


if __name__ == "__main__":
    main()
