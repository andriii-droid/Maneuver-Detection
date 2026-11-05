"""Configuration management for the project."""

from functools import cached_property
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _get_project_root() -> Path:
    return Path(__file__).parents[2]


class PreprocessConfig(BaseModel):
    """Configuration for data preprocessing."""

    block_size: float = Field(
        gt=0, description="Size (in seconds) of the time block for feature summarization"
    )
    stride_size: float = Field(gt=0, description="Stride (in seconds) for moving the time block")


class Config(BaseModel):
    """Main configuration for the project."""

    model_config = ConfigDict(frozen=True)

    project_root: Path = Field(
        default_factory=_get_project_root, description="Root directory of the project"
    )
    data_path: Path = Field(
        default_factory=lambda: _get_project_root() / "data",
        description="Path to the data directory",
    )
    train_data_ids: list[str] = Field(description="List of training data IDs")
    test_data_ids: list[str] = Field(description="List of testing data IDs")
    train_data_path: Path = Field(
        default_factory=lambda: _get_project_root() / "data" / "train.parquet",
        description="Path to training data directory",
    )
    test_data_path: Path = Field(
        default_factory=lambda: _get_project_root() / "data" / "test.parquet",
        description="Path to testing data directory",
    )
    classifiers_path: Path = Field(
        default_factory=lambda: _get_project_root() / "models" / "classifiers.pkl",
        description="Path to save/load classifiers",
    )
    state_description_path: Path = Field(
        default_factory=lambda: _get_project_root() / "data" / "StateDescriptions.json"
    )
    preprocess: PreprocessConfig = Field(description="Configuration for data preprocessing")
    label2id: dict[str, int] = Field(description="Mapping from label names to IDs")
    maneuver_detection_threshold: float = Field(
        0.8, gt=0.0, le=1.0, description="Threshold for maneuver detection accuracy"
    )

    @model_validator(mode="after")
    def _create_directories(self) -> Self:
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.train_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.test_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.classifiers_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_description_path.exists():
            raise FileNotFoundError(
                f"State description file not found at {self.state_description_path}"
            )
        return self

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)

        # Check if user provided necessary preprocess parameters
        preproc = config_dict.get("preprocess", {})
        if preproc.get("block_size") is None:
            raise ValueError(f"preprocess.block_size must be specified in '{yaml_path}'")
        if preproc.get("stride_size") is None:
            raise ValueError(f"preprocess.stride_size must be specified in '{yaml_path}'")

        return cls(**config_dict)

    @cached_property
    def id2label(self) -> dict[int, str]:
        """Mapping from label IDs to names."""
        return {v: k for k, v in self.label2id.items()}

    @cached_property
    def label_names(self) -> list[str]:
        """List of label names."""
        return list(self.label2id.keys())

    @cached_property
    def all_labels(self) -> list[int]:
        """List of all label IDs."""
        return list(self.label2id.values())

    @cached_property
    def background_labels(self) -> list[int]:
        """List of background label IDs."""
        return self.all_labels[0:1]

    @cached_property
    def foreground_labels(self) -> list[int]:
        """List of foreground label IDs."""
        return self.all_labels[1:]


def get_default_config() -> Config:
    """Get the default configuration."""
    root = _get_project_root()
    return Config.from_yaml(root / "configs/default.yml")
