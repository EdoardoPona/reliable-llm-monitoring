"""Generic serialization utilities for experiment results to/from ClearML.

This module provides a reusable framework for serializing any dataclass results
to ClearML and reconstructing them back. It uses dataclass field metadata to
determine how to serialize each field (scalars, artifacts, derived, conditional).

Example:
    @dataclass
    class MyResults:
        accuracy: float = scalar_field()
        confusion_matrix: np.ndarray = artifact_field()
        config: dict = artifact_field()

    serializer = ClearMLSerializer()
    scalars = serializer.to_clearml_scalars(results)
    artifacts = serializer.to_clearml_artifacts(results)
    reconstructed = serializer.from_clearml(task, MyResults)
"""

import logging
from collections.abc import Callable
from dataclasses import Field, field, fields, is_dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# Metadata keys for field configuration
FIELD_TYPE_KEY = "clearml_field_type"
DERIVE_FN_KEY = "clearml_derive_fn"
CONDITION_KEY = "clearml_condition"


def scalar_field(**kwargs) -> Any:
    """Mark a dataclass field as a ClearML scalar metric.

    Scalar fields are logged to ClearML as metrics (float or int values).

    Example:
        @dataclass
        class Results:
            accuracy: float = scalar_field()
            train_size: int = scalar_field()
    """
    metadata = kwargs.pop("metadata", {})
    metadata[FIELD_TYPE_KEY] = "scalar"
    return field(metadata=metadata, **kwargs)


def artifact_field(**kwargs) -> Any:
    """Mark a dataclass field as a ClearML artifact.

    Artifact fields are uploaded to ClearML storage (numpy arrays, dicts, etc).

    Example:
        @dataclass
        class Results:
            predictions: np.ndarray = artifact_field()
            config: dict = artifact_field()
    """
    metadata = kwargs.pop("metadata", {})
    metadata[FIELD_TYPE_KEY] = "artifact"
    return field(metadata=metadata, **kwargs)


def derived_field(derive_fn: Callable, **kwargs) -> Any:
    """Mark a dataclass field as a derived scalar.

    Derived fields are computed on-the-fly from other fields and not stored
    in ClearML. The derive function receives the entire results object.

    Derived fields are excluded from __init__ since they're computed, not provided.

    Example:
        @dataclass
        class Results:
            predictions: np.ndarray = artifact_field()
            mean_prediction: float = derived_field(
                derive_fn=lambda r: float(r.predictions.mean())
            )
    """
    metadata = kwargs.pop("metadata", {})
    metadata[FIELD_TYPE_KEY] = "derived"
    metadata[DERIVE_FN_KEY] = derive_fn
    # Derived fields don't go in __init__ and have default value
    # (won't be used, but needed to prevent required positional arg error)
    return field(metadata=metadata, init=False, default=None, **kwargs)


def conditional_field(condition: str, **kwargs) -> Any:
    """Mark a dataclass field as conditional.

    Conditional fields are only serialized when a boolean condition field
    is True. Useful for optional test results that only exist on success.

    Args:
        condition: Name of the boolean field that gates this field's serialization.

    Example:
        @dataclass
        class Results:
            success: bool = scalar_field()
            best_score: float | None = conditional_field(condition="success")
    """
    metadata = kwargs.pop("metadata", {})
    metadata[CONDITION_KEY] = condition
    return field(metadata=metadata, **kwargs)


@runtime_checkable
class FieldSerializer(Protocol):
    """Protocol for custom field serializers.

    Allows extending the serialization logic for specific field types
    beyond the built-in scalars, artifacts, derived, and conditional types.
    """

    def should_handle(self, field_value: Any, field_obj: Field) -> bool:
        """Check if this serializer should handle the field.

        Args:
            field_value: Current value of the field
            field_obj: Dataclass Field object

        Returns:
            True if this serializer should handle this field
        """
        ...

    def serialize_to_scalar(self, field_value: Any, field_obj: Field) -> float | int | None:
        """Serialize field value to a scalar.

        Args:
            field_value: Current value of the field
            field_obj: Dataclass Field object

        Returns:
            Scalar value (float or int) or None if not applicable
        """
        ...

    def serialize_to_artifact(self, field_value: Any, field_obj: Field) -> Any | None:
        """Serialize field value to an artifact.

        Args:
            field_value: Current value of the field
            field_obj: Dataclass Field object

        Returns:
            Artifact data or None if not applicable
        """
        ...

    def deserialize(self, value: Any, field_obj: Field) -> Any:
        """Deserialize value back to original type.

        Args:
            value: Serialized value
            field_obj: Dataclass Field object

        Returns:
            Deserialized value
        """
        ...


class ClearMLSerializer:
    """Generic serializer for dataclass results to/from ClearML.

    This serializer works with any dataclass that uses the field metadata
    helpers (scalar_field, artifact_field, derived_field, conditional_field)
    to determine how to serialize/deserialize each field.

    It handles:
    - Extracting scalar metrics for ClearML logging
    - Extracting artifacts (numpy arrays, configs) for ClearML storage
    - Reconstructing dataclass instances from ClearML tasks
    - Conditional fields (only serialized when a condition is True)
    - Derived fields (computed on-the-fly, not stored)
    """

    def __init__(self):
        """Initialize the serializer."""
        self._custom_serializers: list[FieldSerializer] = []

    def register_custom_serializer(self, serializer: FieldSerializer) -> None:
        """Register a custom field serializer for specialized handling.

        Args:
            serializer: Custom serializer implementing FieldSerializer protocol
        """
        self._custom_serializers.append(serializer)

    def to_clearml_scalars(self, results: Any, include_derived: bool = True) -> dict[str, float | int]:
        """Extract scalar metrics from results dataclass.

        Args:
            results: Results dataclass instance
            include_derived: Whether to compute and include derived fields

        Returns:
            Dictionary of scalar metrics suitable for ClearML logging

        Raises:
            TypeError: If results is not a dataclass
        """
        if not is_dataclass(results):
            raise TypeError(f"Expected dataclass instance, got {type(results)}")

        scalars = {}

        for field_obj in fields(results):
            field_value = getattr(results, field_obj.name)
            field_meta = field_obj.metadata

            # Check conditional fields
            if CONDITION_KEY in field_meta:
                condition_field = field_meta[CONDITION_KEY]
                if not getattr(results, condition_field, False):
                    continue  # Skip this field

            # Get field type
            field_type = field_meta.get(FIELD_TYPE_KEY, self._infer_field_type(field_value))

            # Handle derived fields
            if field_type == "derived":
                if include_derived:
                    derive_fn = field_meta.get(DERIVE_FN_KEY)
                    if derive_fn:
                        try:
                            scalars[field_obj.name] = derive_fn(results)
                        except Exception as e:
                            logger.warning(f"Failed to compute derived field {field_obj.name}: {e}")
                continue

            # Handle scalar fields
            if field_type == "scalar":
                scalar_value = self._to_scalar(field_value, field_obj)
                if scalar_value is not None:
                    scalars[field_obj.name] = scalar_value

        return scalars

    def to_clearml_artifacts(self, results: Any) -> dict[str, Any]:
        """Extract artifacts from results dataclass.

        Args:
            results: Results dataclass instance

        Returns:
            Dictionary of artifacts suitable for ClearML logging (numpy arrays, dicts, etc)

        Raises:
            TypeError: If results is not a dataclass
        """
        if not is_dataclass(results):
            raise TypeError(f"Expected dataclass instance, got {type(results)}")

        artifacts = {}

        for field_obj in fields(results):
            field_value = getattr(results, field_obj.name)
            field_meta = field_obj.metadata

            # Check conditional fields
            if CONDITION_KEY in field_meta:
                condition_field = field_meta[CONDITION_KEY]
                if not getattr(results, condition_field, False):
                    continue

            # Get field type
            field_type = field_meta.get(FIELD_TYPE_KEY, self._infer_field_type(field_value))

            # Handle artifact fields
            if field_type == "artifact":
                artifacts[field_obj.name] = field_value

            # Exclude derived and scalar fields from artifacts
            if field_type in ("derived", "scalar", "exclude"):
                continue

        return artifacts

    def from_clearml(self, task: Any, result_class: type) -> Any:
        """Reconstruct results dataclass from ClearML task.

        Args:
            task: ClearML Task object
            result_class: The dataclass type to reconstruct

        Returns:
            Reconstructed results instance

        Raises:
            TypeError: If result_class is not a dataclass
            ValueError: If configuration cannot be loaded from task
        """
        if not is_dataclass(result_class):
            raise TypeError(f"Expected dataclass type, got {type(result_class)}")

        # Fetch configuration (handles both ClearML 1.x and 2.x)
        config_dict = self._fetch_configuration(task)

        # Fetch artifacts
        artifacts = task.artifacts if hasattr(task, "artifacts") else {}

        # Fetch scalars
        reported_scalars = self._get_reported_scalars(task)
        results_section = reported_scalars.get("Results", {})

        # Build field values
        field_values = {}

        for field_obj in fields(result_class):
            field_meta = field_obj.metadata
            field_type = field_meta.get(FIELD_TYPE_KEY, self._infer_field_type_from_annotation(field_obj.type))

            # Handle conditional fields - check condition first
            if CONDITION_KEY in field_meta:
                condition_field = field_meta[CONDITION_KEY]
                # Get condition value from already-loaded fields or scalars
                condition_value = field_values.get(condition_field)
                if condition_value is None:
                    # Try to fetch from scalars
                    condition_value = self._get_scalar_from_task(results_section, condition_field, False)

                if not condition_value:
                    field_values[field_obj.name] = None
                    continue

            # Load field value based on type
            if field_type == "scalar":
                field_values[field_obj.name] = self._get_scalar_from_task(
                    results_section, field_obj.name, self._get_default_value(field_obj.type)
                )

            elif field_type == "artifact":
                if field_obj.name in artifacts:
                    field_values[field_obj.name] = self._load_artifact(artifacts[field_obj.name], field_obj.type)
                else:
                    field_values[field_obj.name] = None

            elif field_type == "derived":
                # Derived fields are computed, not stored - use default value
                field_values[field_obj.name] = self._get_default_value(field_obj.type)

            else:
                # Try config first, then scalars, then default
                if field_obj.name in config_dict:
                    field_values[field_obj.name] = config_dict[field_obj.name]
                else:
                    field_values[field_obj.name] = self._get_scalar_from_task(
                        results_section,
                        field_obj.name,
                        self._get_default_value(field_obj.type),
                    )

        return result_class(**field_values)

    # Helper methods

    def _infer_field_type(self, value: Any) -> str:
        """Infer field type from value (convention-based fallback)."""
        if value is None:
            return "exclude"
        if isinstance(value, np.ndarray):
            return "artifact"
        if isinstance(value, dict):
            return "artifact"
        if isinstance(value, (int, float, bool)):
            return "scalar"
        return "exclude"

    def _infer_field_type_from_annotation(self, annotation: type | str) -> str:
        """Infer field type from type annotation."""
        from typing import Union, get_args, get_origin

        # Handle Optional types
        origin = get_origin(annotation)
        if origin is Union:
            # Get non-None type
            args = [a for a in get_args(annotation) if a is not type(None)]
            if args:
                annotation = args[0]

        if annotation is dict or get_origin(annotation) is dict:
            return "artifact"
        if annotation is np.ndarray:
            return "artifact"
        if annotation in (int, float, bool):
            return "scalar"
        return "exclude"

    def _to_scalar(self, value: Any, field_obj: Field) -> float | int | None:
        """Convert value to scalar."""
        # Try custom serializers first
        for serializer in self._custom_serializers:
            if serializer.should_handle(value, field_obj):
                result = serializer.serialize_to_scalar(value, field_obj)
                if result is not None:
                    return result

        # Handle standard types
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return value

        return None

    def _fetch_configuration(self, task: Any) -> dict:
        """Fetch configuration from ClearML task (handles 1.x and 2.x)."""
        import yaml

        config_dict = None

        # Try ClearML 2.x method
        if hasattr(task, "get_configuration_object_as_dict"):
            try:
                config_dict = task.get_configuration_object_as_dict("Configuration")
            except Exception:
                config_dict = None

        # Try ClearML 1.x method
        if config_dict is None and hasattr(task, "get_parameter"):
            try:
                config_dict = task.get_parameter("Configuration")
            except Exception:
                config_dict = None

        # Handle YAML-encoded config
        if isinstance(config_dict, str):
            try:
                config_dict = yaml.safe_load(config_dict)
            except Exception as e:
                logger.warning(f"Failed to parse configuration as YAML: {e}")
                config_dict = None

        # Fallback: config artifact
        if config_dict is None and "config" in task.artifacts:
            try:
                config_path = task.artifacts["config"].get_local_copy()
                with open(config_path) as f:
                    config_dict = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load configuration from artifact: {e}")
                config_dict = None

        if config_dict is None:
            raise ValueError("Could not load configuration from ClearML task")

        return dict(config_dict) if not isinstance(config_dict, dict) else config_dict

    def _get_reported_scalars(self, task: Any) -> dict:
        """Get reported scalars from ClearML task."""
        try:
            if hasattr(task, "get_logger"):
                logger_obj = task.get_logger()
                if hasattr(logger_obj, "get_metrics"):
                    return logger_obj.get_metrics()
        except Exception:
            pass
        return {}

    def _get_scalar_from_task(self, results_section: dict, name: str, default: Any) -> Any:
        """Extract scalar value from ClearML results section."""
        series = results_section.get(name)
        if series is None:
            return default

        # Handle dict format {y: [values]}
        if isinstance(series, dict) and "y" in series:
            y = series.get("y")
            if isinstance(y, list) and len(y) > 0:
                return y[-1]
            return default

        # Handle list format [values]
        if isinstance(series, list) and len(series) > 0:
            return series[-1]

        # Handle scalar value
        if isinstance(series, (int, float, bool)):
            return series

        return default

    def _load_artifact(self, artifact: Any, expected_type: type | str) -> Any:
        """Load artifact from ClearML."""
        try:
            local_path = artifact.get_local_copy()

            # Handle numpy arrays
            if expected_type is np.ndarray or "ndarray" in str(expected_type):
                return np.load(local_path)

            # Handle dicts (YAML)
            if expected_type is dict:
                import yaml

                with open(local_path) as f:
                    return yaml.safe_load(f)

            return local_path
        except Exception as e:
            logger.warning(f"Failed to load artifact: {e}")
            return None

    def _get_default_value(self, type_annotation: type | str) -> Any:
        """Get default value for type."""
        from typing import get_origin

        if type_annotation is int:
            return 0
        if type_annotation is float:
            return 0.0
        if type_annotation is bool:
            return False
        if type_annotation is str:
            return ""
        if type_annotation is np.ndarray:
            return None
        if type_annotation is dict:
            return None
        # For Optional types, return None
        if get_origin(type_annotation) is type(None):
            return None
        return None
