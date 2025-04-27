from setuptools import setup, find_packages

setup(
    name="efficientvit",
    version="0.1.0",
    description="EfficientViT: A Lightweight Vision Transformer from Microsoft Research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Microsoft Research",
    author_email="N/A",
    url="https://github.com/microsoft/Cream/tree/main/EfficientViT",
    packages=find_packages(include=["efficientvit*"]),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "timm>=0.6.0",
        "numpy",
        "pillow",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)