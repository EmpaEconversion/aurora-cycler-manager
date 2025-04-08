"""Version information for the Aurora cycler manager package."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import toml


def get_version() -> str:
    """Get version information."""
    try:
        return version("aurora-cycler-manager")
    except PackageNotFoundError:
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        with pyproject_path.open("r") as f:
            return toml.load(f)["project"]["version"] + "-dev"


__version__ = get_version()
__author__ = "Graham Kimbell"
__title__ = "Aurora cycler manager"
__license__ = "MIT"
__copyright__ = "2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia"
__description__ = """
Tools to distribute jobs, monitor progress and analyse results from Tomato
battery cycling servers.
"""
__url__ = "https://github.com/EmpaEconversion/aurora-cycler-manager"
