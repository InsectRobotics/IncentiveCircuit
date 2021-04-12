import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="incentive-circuit",
    version="v1.0.0-alpha",
    author="Evripidis Gkanias",
    maintainer="Evripidis Gkanias",
    author_email="ev.gkanias@ed.ac.uk",
    maintainer_email="ev.gkanias@ed.ac.uk",
    description="A package implementing the incentive circuit in the fruit fly brain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evgkanias/IncentiveCircuit",
    project_urls={
        "Bug Tracker": "https://github.com/evgkanias/IncentiveCircuit/issues"
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Licence :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent"
    ],
    packages=["incentive"],
    package_dir={"": "src"},
    data_files=[('data/FruitflyMB', ['data/FruitflyMB/meta.yaml'])],
    python_requires=">=3.7",
)
