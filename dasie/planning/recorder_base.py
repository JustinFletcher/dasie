"""
Logging base class for training runs. The philosophy is that each repo will require its
own unique analytics and thus is to inherit the RecorderBase and implement any necessary
custom logic. The base class here provides many of the common utilities and methods
shared among different ML projects.
"""
from functools import partial
import io
import collections
import logging
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Callable, List, Tuple, Union

import PIL
import matplotlib
from matplotlib import pyplot as plt
import mlflow
import numpy as np
import pyrallis


# use Agg backend for image rendering to file
matplotlib.use("Agg")


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] Recorder (%(levelname)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class AsyncCaller:
    """
    Implements asynchronous execution of class methods within an arbitrary class by
    providing a convenient decorator `async_dec`. Uses a single queue and multiplethreads
    (workers) to consume the queue (i.e. consumer/producer pattern).

    Usage:
        Decorate a class method via `@AsyncCaller.async_dec(ac_attr="async_log")`. The
        target class must possess the corresponding attribute `async_log = AsyncCaller()`.

    Note:
        In applying the decorator @AsyncCaller.async_dec(ac_attr="async_log"), the
        async_dec method will look at the class instance (e.g. Recoder or RecorderBase)
        and determine if there is an attribute named 'async_log' and if that attribute is
        callable. If true, serves as the indicator that async calling is to be applied.

        Additionally, the `wrapper` function is only called when the decorated method is
        called, otherwise the decorated methods simply point to the `wrapper` function.
    """

    STOP_MARK = "__STOP"

    def __init__(self, num_threads=4) -> None:
        self._q = Queue()
        self._stop = False
        self.num_threads = num_threads
        self.threads = []
        self.start_threads()

    def start_threads(self):
        """spin up n threads actively consuming queue via `run`"""
        for _ in range(self.num_threads):
            t = Thread(target=self.run)
            self.threads.append(t)
            t.start()

    def close(self):
        """add STOP_MARK to queue to trigger thread termination"""
        self._q.put(self.STOP_MARK)

    def run(self):
        """consume queue until STOP_MARK is reached"""
        while True:
            data = self._q.get()
            if data == self.STOP_MARK:
                break
            data()

    def __call__(self, func, *args, **kwargs):
        self._q.put(partial(func, *args, **kwargs))

    def wait(self, close=True):
        """block parent thread until async threads terminate"""
        if close:
            for t in self.threads:
                self.close()
        for i, t in enumerate(self.threads):
            logger.info(f"closing thread {i + 1}/{self.num_threads}")
            logger.info(f"current queue length: {self._q.qsize()}")
            t.join()
            logger.info(f"thread {i + 1}/{self.num_threads} closed")

    @staticmethod
    def async_dec(ac_attr):
        def decorator_func(func):
            def wrapper(self, *args, **kwargs):
                if isinstance(getattr(self, ac_attr, None), Callable):
                    return getattr(self, ac_attr)(func, self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)

            return wrapper

        return decorator_func


class RecorderBase:
    """
    To be inherited by custom user-defined Recorder class. Provides core backend methods for
    managing runs/experiments and assists in fine control of logging training run quantities.
    Includes a number of image logging helper utilities to be called within child class.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.log.uri is not None:
            self.uri = str(Path(cfg.log.uri).expanduser())
            mlflow.set_tracking_uri(self.uri)
        else:
            self.uri = mlflow.get_tracking_uri()
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self.uri)
        self.exp_id = None
        self.run = None
        self.run_id = None
        self._artifact_uri = None
        self.root = None
        self.async_log = None

        self.curr_step = 0

    def set_experiment(self, exp_name=None):
        """set exp_id as attribute, assumes experiment already exists"""
        if self.cfg.log.exp_name is not None:
            self.exp_id = mlflow.get_experiment_by_name(
                self.cfg.log.exp_name
            ).experiment_id
        else:
            self.exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
        return self.exp_id

    def set_run(self, run=None):
        """set run and run_id attributes, assuming run already exists"""
        self.run = mlflow.active_run() if run is None else run
        self.run_id = self.run.info.run_id
        self._artifact_uri = self.run.info.artifact_uri
        if "file:/" in self._artifact_uri:
            self._artifact_uri = self.run.info.artifact_uri[7:]
        self.root = Path(self._artifact_uri)  # cut file:/ uri
        logger.info(f"starting mlflow run: {Path(self.root).parent}")

        self.async_log = AsyncCaller() if self.cfg.log.enable_async else None
        return self.run

    def create_experiment(self):
        # create experiment if specified in cfg and does not exist
        if self.cfg.log.exp_name is not None:
            if mlflow.get_experiment_by_name(self.cfg.log.exp_name) is None:
                logger.info(f"creating mlflow experiment: {self.cfg.log.exp_name}")
                self.exp_id = mlflow.create_experiment(self.cfg.log.exp_name)
            else:
                self.exp_id = mlflow.get_experiment_by_name(
                    self.cfg.log.exp_name
                ).experiment_id
        return self.exp_id

    def start_run(self):
        # start run
        self.run = mlflow.start_run(
            run_id=None, experiment_id=self.exp_id, run_name=self.cfg.log.run_name
        )

        # save the run id and artifact_uri
        self.run_id = self.run.info.run_id
        self._artifact_uri = self.run.info.artifact_uri
        if "file:/" in self._artifact_uri:
            self._artifact_uri = self.run.info.artifact_uri[7:]
        self.root = Path(self._artifact_uri)  # cut file:/ uri
        logger.info(f"starting mlflow run: {Path(self.root).parent}")

        # initialize async logging
        if self.cfg.log.enable_async:
            n_threads = self.cfg.log.num_threads
            logger.info(f"enabling async logging with {n_threads} threads")
            # note matching "async_log" in AsyncCaller method decorations
            self.async_log = AsyncCaller(num_threads=n_threads)
        else:
            self.async_log = None
        return self.run

    def end_run(self):
        mlflow.end_run()
        if self.async_log is not None:
            logger.info("waiting for recorder threads to finish")
            self.async_log.wait()
        self.async_log = None

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_artifact(self, local_path, artifact_path=None):
        self.client.log_artifact(self.run_id, local_path, artifact_path)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_dict(self, d, artifact_path=None):
        self.client.log_dict(self.run_id, d, artifact_path)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_metric(self, k, v, step=None):
        self.client.log_metric(self.run_id, k, v, step=step)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_metrics(self, d, step=None):
        for name, data in d.items():
            if data is not None:
                self.client.log_metric(self.run_id, name, data, step=step)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_params(self, param_dict):
        for name, data in param_dict.items():
            self.client.log_param(self.run_id, name, data)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_text(self, text, artifact_path=None):
        self.client.log_text(self.run_id, text, artifact_path)

    @AsyncCaller.async_dec(ac_attr="async_log")
    def log_image(self, image: Union[np.ndarray, PIL.Image.Image], artifact_path=None):
        self.client.log_image(self.run_id, image, artifact_path)

    def figure_to_array(
        self, figure: plt.Figure, png: bool = True, bbox_inches=None
    ) -> np.ndarray:
        """convert matplotlib figure to numpy array"""
        fmt = "png" if png else "jpg"
        buffer = io.BytesIO()
        figure.savefig(buffer, format=fmt, dpi=100, bbox_inches=bbox_inches)
        buffer.seek(0)
        img = PIL.Image.open(buffer)
        data = np.array(img)
        plt.close(figure)
        return data

    def log_run_params(self, name="train_cfg.yaml"):
        """Log config dataclass as params for this run"""
        # cfg as dict, encoded for yaml
        cfg_dict = pyrallis.encode(self.cfg)
        self.log_dict(cfg_dict, f"archive/{name}")
        self.log_params(self.flatten(cfg_dict))

    @staticmethod
    def flatten(d, parent_key="", sep="."):
        """Flatten a nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(RecorderBase.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)