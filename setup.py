"""
Setup script for Heightcraft.

This script installs the Heightcraft package and its dependencies.
"""

from setuptools import find_packages, setup

# Read requirements from file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read long description from README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="heightcraft",
    version="1.0.0",
    description="A powerful and flexible height map generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Heightcraft Team",
    author_email="info@heightcraft.com",
    url="https://github.com/heightcraft/heightcraft",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "heightcraft=heightcraft.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="height map, 3D model, visualization, rendering, terrain",
) 