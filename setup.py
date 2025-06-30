from setuptools import setup, find_packages

setup(
    name="functa_torch",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "functa-train=functa_torch.run_experiment_with_args:main",
        ],
    },
)
