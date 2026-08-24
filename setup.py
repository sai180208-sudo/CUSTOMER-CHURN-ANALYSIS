"""
Setup configuration for Customer Churn Prediction project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="customer-churn-prediction",
    version="1.0.0",
    author="Data Science Team",
    author_email="your.email@example.com",
    description="ML pipeline for predicting customer churn using classification algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/customer-churn-prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "notebook>=6.0.0",
        ],
    },
    keywords="churn prediction machine-learning classification scikit-learn",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/customer-churn-prediction/issues",
        "Source Code": "https://github.com/yourusername/customer-churn-prediction",
        "Documentation": "https://github.com/yourusername/customer-churn-prediction#readme",
    },
)
