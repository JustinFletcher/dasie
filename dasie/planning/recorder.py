"""User defined logging and analytics utility for training runs"""
import logging
from pathlib import Path

from recorder_base import RecorderBase

logger = logging.getLogger(__name__)


class Recorder(RecorderBase):
    """Artifact, metric, parameter, and image logger. Define custom analytic logic here."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def log_files(self):
        """log relevant source files to mlflow"""
        # directory of this recorder.py script
        script_dir = Path(__file__).parent
        src_files = [
            "differentiable_dasie.py",
            "get_plan.py",
            "plot_differentiable_dasie.py",
            "recorder.py",
            "recorder_base.py",
            "recover.py",
            "recovery_models.py",
            "train.py",
        ]
        # log them to /archive mlflow sub directory
        for relpath in src_files:
            self.log_artifact(script_dir / relpath, "archive")

    def custom_figure_creation_method(self):
        raise NotImplementedError

    def custom_metric_computation_method(self):
        raise NotImplementedError

