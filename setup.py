"""
Setup script for Sokuji-Bridge

This setup.py allows all modules under src/ to be imported as top-level packages.
For example: from config.manager import ConfigManager
"""

from setuptools import setup, find_packages

# find_packages will discover: config, core, providers, utils, api, services, cli
packages = find_packages(where="src")

setup(
    name="sokuji-bridge",
    packages=packages,
    package_dir={"": "src"},
)
