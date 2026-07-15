"""
Project configuration.

All project paths are defined here.
"""

from pathlib import Path


# =============================================================================
# Project root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Data directories
# =============================================================================

DATASETS = PROJECT_ROOT / "datasets"

RAW_DATA = DATASETS / "raw"

PROCESSED_DATA = DATASETS / "processed"


# =============================================================================
# Outputs
# =============================================================================

OUTPUTS = PROJECT_ROOT / "outputs"

TABLES = OUTPUTS / "tables"

FIGURES = OUTPUTS / "figures"

PREDICTIONS = OUTPUTS / "predictions"

LOGS = OUTPUTS / "logs"


# =============================================================================
# Models
# =============================================================================

MODELS = PROJECT_ROOT / "models"


# =============================================================================
# Create directories if needed
# =============================================================================

DIRECTORIES = [

    RAW_DATA,
    PROCESSED_DATA,

    OUTPUTS,
    TABLES,
    FIGURES,
    PREDICTIONS,
    LOGS,

    MODELS

]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)