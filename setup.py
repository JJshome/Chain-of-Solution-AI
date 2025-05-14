from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="chain-of-solution",
    version="0.1.0",
    author="Jee Hwan Jang",
    author_email="jeehwan.jang@skku.edu",
    description="An AI framework integrating multiple problem-solving methodologies with large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JJshome/Chain-of-Solution-AI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
