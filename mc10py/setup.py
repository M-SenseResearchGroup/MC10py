import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="MC10py",
    version="0.0.1",
    author="Lukas Adamowicz",
    author_email="lukas.adamowicz95@gmail.com",
    description="Import and segment raw MC10 BioStamp data",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/M-SenseResearchGroup/MC10py",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPL 3.0",
        "Operating System :: OS Independent",
    ),
)
