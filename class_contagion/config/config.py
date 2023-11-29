from pathlib import Path

import class_contagion

# Project Directories
PACKAGE_ROOT = Path(class_contagion.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
MODEL_DIR = PACKAGE_ROOT / "model"
PROCESS_DIR = PACKAGE_ROOT / "process"
