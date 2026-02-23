"""ClearML experiment tracking and logging utilities."""

import logging
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ClearMLLogger:
    """Optional ClearML task logger with graceful error handling.

    Enables experiment tracking without disrupting the experiment itself.
    If ClearML is not available or enabled=False, all methods are no-ops.
    """

    def __init__(self, project_name: str, task_name: str, enabled: bool = True):
        """Initialize ClearML logger with timestamped task name.

        Args:
            project_name: ClearML project name
            task_name: Base task name (timestamp will be appended)
            enabled: Whether to actually use ClearML
        """
        self.enabled = enabled
        self.task = None
        self.temp_dir = None  # Keep temp directory until finalize() is called

        if not enabled:
            return

        try:
            from clearml import Task

            # Add timestamp to task name for uniqueness
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            timestamped_name = f"{task_name}_{timestamp}"

            # Initialize ClearML task
            # Disable auto-detection of PyTorch and Transformers to avoid duplicate
            # model warnings when loading the same models multiple times
            self.task = Task.init(
                project_name=project_name,
                task_name=timestamped_name,
                auto_connect_frameworks={
                    "pytorch": False,
                    "transformers": False,
                },
            )
            logger.info(f"Initialized ClearML task: {project_name} / {timestamped_name}")
        except ImportError:
            warnings.warn(
                "ClearML not installed. Experiment tracking disabled. Install with: pip install clearml",
                UserWarning,
                stacklevel=2,
            )
            self.enabled = False
        except Exception as e:
            warnings.warn(
                f"Failed to initialize ClearML task: {e}. Experiment will continue without tracking.",
                UserWarning,
                stacklevel=2,
            )
            self.enabled = False

    def connect_configuration(self, config: dict) -> None:
        """Log configuration parameters to ClearML.

        Args:
            config: Configuration dictionary to log
        """
        if not self.enabled or self.task is None:
            return

        try:
            # Log config as parameters with proper nesting
            self.task.connect_configuration(
                configuration=config,
                name="Configuration",
            )
            logger.info("Logged configuration to ClearML")
        except Exception as e:
            warnings.warn(
                f"Failed to log configuration to ClearML: {e}",
                UserWarning,
                stacklevel=2,
            )

    def add_tags(self, tags: list[str]) -> None:
        """Add tags to the task for organization and filtering.

        Args:
            tags: List of tag strings to add
        """
        if not self.enabled or self.task is None:
            return

        try:
            self.task.add_tags(tags)
            logger.info(f"Added tags to ClearML task: {tags}")
        except Exception as e:
            warnings.warn(
                f"Failed to add tags to ClearML task: {e}",
                UserWarning,
                stacklevel=2,
            )

    def log_scalars(self, scalars: dict[str, float | int]) -> None:
        """Log scalar metrics to ClearML.

        Args:
            scalars: Dictionary of metric_name -> value pairs
        """
        if not self.enabled or self.task is None:
            return

        try:
            logger_instance = self.task.get_logger()
            for name, value in scalars.items():
                # Log to "Results" section
                logger_instance.report_scalar(
                    title="Results",
                    series=name,
                    value=float(value),
                    iteration=0,
                )
            logger.info(f"Logged {len(scalars)} scalar metrics to ClearML")
        except Exception as e:
            warnings.warn(
                f"Failed to log scalars to ClearML: {e}",
                UserWarning,
                stacklevel=2,
            )

    def log_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Log artifacts (numpy arrays and config) to ClearML.

        Args:
            artifacts: Dictionary of artifact_name -> data pairs
                      (numpy arrays or dicts)
        """
        if not self.enabled or self.task is None:
            return

        try:
            # Create temporary directory for artifacts
            # Keep reference to delete it later in finalize()
            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp()
            temp_path = Path(self.temp_dir)

            # Save numpy arrays and upload
            for name, data in artifacts.items():
                if isinstance(data, np.ndarray):
                    file_path = temp_path / f"{name}.npy"
                    np.save(file_path, data)
                    self.task.upload_artifact(
                        name=name,
                        artifact_object=str(file_path),
                    )
                elif isinstance(data, dict):
                    # Save config as YAML
                    import yaml

                    file_path = temp_path / f"{name}.yaml"
                    with open(file_path, "w") as f:
                        yaml.dump(data, f)
                    self.task.upload_artifact(
                        name=name,
                        artifact_object=str(file_path),
                    )

            logger.info(f"Logged {len(artifacts)} artifacts to ClearML")
        except Exception as e:
            warnings.warn(
                f"Failed to log artifacts to ClearML: {e}",
                UserWarning,
                stacklevel=2,
            )

    def log_pickle_artifact(self, name: str, obj: Any) -> None:
        """Upload an arbitrary Python object as a pickle artifact.

        Unlike :meth:`log_artifacts` (which only handles numpy arrays and
        dicts), this method pickles *any* Python object -- useful for saving
        full experiment result dataclasses for later retrieval.

        Args:
            name: Artifact name (used to retrieve it later).
            obj: Any pickle-able Python object.
        """
        if not self.enabled or self.task is None:
            return

        try:
            import pickle

            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp()
            file_path = Path(self.temp_dir) / f"{name}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
            self.task.upload_artifact(name=name, artifact_object=str(file_path))
            logger.info(f"Uploaded pickle artifact '{name}' to ClearML")
        except Exception as e:
            warnings.warn(
                f"Failed to upload pickle artifact '{name}' to ClearML: {e}",
                UserWarning,
                stacklevel=2,
            )

    def log_figure(self, title: str, series: str, figure, iteration: int = 0) -> None:
        """Log a matplotlib figure to ClearML.

        Args:
            title: Plot title/group (e.g., "Comparison", "Distributions")
            series: Plot series name (e.g., "Summary Statistics", "budget_cost")
            figure: Matplotlib figure object
            iteration: Iteration number (default 0)
        """
        if not self.enabled or self.task is None:
            return

        try:
            logger_instance = self.task.get_logger()
            logger_instance.report_matplotlib_figure(
                title=title,
                series=series,
                figure=figure,
                iteration=iteration,
            )
            logger.info(f"Logged figure '{title}/{series}' to ClearML")
        except Exception as e:
            warnings.warn(
                f"Failed to log figure to ClearML: {e}",
                UserWarning,
                stacklevel=2,
            )

    def finalize(self) -> None:
        """Close and finalize the ClearML task, cleaning up temporary artifacts."""
        if not self.enabled or self.task is None:
            return

        try:
            self.task.close()
            logger.info("ClearML task finalized")
        except Exception as e:
            warnings.warn(
                f"Failed to finalize ClearML task: {e}",
                UserWarning,
                stacklevel=2,
            )
        finally:
            # Clean up temporary directory after task close
            if self.temp_dir is not None:
                try:
                    shutil.rmtree(self.temp_dir)
                    logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary directory {self.temp_dir}: {cleanup_error}")
