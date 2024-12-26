from setuptools import setup, find_packages

setup(
    name="warp-speed-dataset",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "datasets>=2.0.0",
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "pandas>=1.3.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "networkx>=2.6.0",
        "nltk>=3.6.0",
    ],
    author="GotThatData",
    author_email="contact@gotthatdata.com",
    description="A comprehensive dataset for warp drive research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GotThatData/warp-speed-dataset",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
)