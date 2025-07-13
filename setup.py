from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "A library for federated learning and personalized federated learning with Generalized Linear Models"

# Read requirements
def read_requirements():
    try:
        with open('requirements.txt') as req:
            return [line.strip() for line in req if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        # Fallback requirements if file doesn't exist
        return [
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "statsmodels>=0.13.0",
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0"
        ]

setup(
    name="federated-glm",
    version="0.2.0",
    author="Mohammad Amini",
    author_email="m.amini@ufl.edu",
    description="A library for federated learning and personalized federated learning with Generalized Linear Models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mhmdamini/federated-glm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",  # Changed from Alpha to Beta
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "sphinx",
            "sphinx-rtd-theme",
        ],
        "examples": [
            "matplotlib",
            "seaborn",
            "jupyter",
        ]
    },
    include_package_data=True,
    zip_safe=False,
    keywords=["federated-learning", "glm", "machine-learning", "statistics", "personalization"],
    project_urls={
        "Bug Reports": "https://github.com/mhmdamini/federated-glm/issues",
        "Source": "https://github.com/mhmdamini/federated-glm",
        "Documentation": "https://github.com/mhmdamini/federated-glm#readme",
    },
)