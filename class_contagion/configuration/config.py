from pathlib import Path

import configuration

# Project Directories
PACKAGE_ROOT = Path(configuration.__file__).resolve().parent.parent
ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PACKAGE_ROOT / "configuration"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
MODEL_DIR = PACKAGE_ROOT / "model"
PROCESS_DIR = PACKAGE_ROOT / "process"
