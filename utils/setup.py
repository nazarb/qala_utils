"""
Setup script for qanat detection package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Qanat Detection Pipeline - Automated detection from satellite imagery"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="qanat-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated qanat detection from satellite imagery using YOLO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qanat-detection",
    package_dir={"": ".."},
    packages=["qala_processor"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ]
    },
    entry_points={
        "console_scripts": [
            "qanat-detect=qala_processor.run_detection:main",
        ],
    },
    include_package_data=True,
    keywords="qanat archaeology satellite-imagery object-detection yolo machine-learning gis",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/qanat-detection/issues",
        "Source": "https://github.com/yourusername/qanat-detection",
        "Documentation": "https://github.com/yourusername/qanat-detection#readme",
    },
)
