from setuptools import setup, find_packages

setup(
    name="tt-spin-tracker",
    version="1.0.0",
    author="Your Name",
    description="Table tennis ball trajectory and spin detection system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "tt-track=src.processing.tracker:main",
            "tt-train=src.modeling.train_classifier:main",
            "tt-serve=src.api.main:start",
        ]
    },
)
