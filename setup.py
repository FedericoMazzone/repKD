import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="repkd",
    version="0.0.1",
    author="Federico Mazzone et al.",
    author_email="f.mazzone@utwente.nl",
    description="Repeated Knowledge Distillation with Confidence Masking to "
                "Mitigate Membership Inference Attacks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FedericoMazzone/repKD",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
