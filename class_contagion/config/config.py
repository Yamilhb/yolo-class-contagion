from pathlib import Path

import config

# Project Directories
PACKAGE_ROOT = Path(config.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
#CONFIG_DIR = PACKAGE_ROOT / "config"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
MODEL_DIR = PACKAGE_ROOT / "model"
PROCESS_DIR = PACKAGE_ROOT / "process"
