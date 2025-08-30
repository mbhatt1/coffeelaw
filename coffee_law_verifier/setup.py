"""
Setup script for Coffee Law Verifier
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="coffee-law-verifier",
    version="1.0.0",
    author="Coffee Law Research Team",
    author_email="coffee-law@example.com",
    description="Production Monte Carlo simulation for Coffee Law verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/coffee-law/verifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "coffee-law-verify=coffee_law_verifier.run_verification:main",
            "coffee-law-dashboard=coffee_law_verifier.visualization.interactive_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "coffee_law_verifier": ["data/*.json", "templates/*.html"],
    },
)