from setuptools import setup, find_packages

setup(
    name="graspnet_baseline",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "tensorboard",
        "numpy",
        "scipy",
        "open3d>=0.8",
        "Pillow",
        "tqdm",
    ],
)
