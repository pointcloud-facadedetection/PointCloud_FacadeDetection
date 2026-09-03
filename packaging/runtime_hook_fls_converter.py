"""Locate the bundled FlsConverter package beside the PyInstaller exe."""
import os
import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    root = Path(sys.executable).resolve().parent
else:
    root = Path(__file__).resolve().parents[1]

converter_dir = root / "FlsConverter_package" / "FlsConverter"
if converter_dir.is_dir():
    os.environ["FLS_CONVERTER_DLL_DIR"] = str(converter_dir)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(converter_dir))
    os.environ["PATH"] = str(converter_dir) + os.pathsep + os.environ.get("PATH", "")
    if str(converter_dir) not in sys.path:
        sys.path.insert(0, str(converter_dir))