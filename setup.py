import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="incentive-circuit",
    version="1.0.0",
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
        "Programming Language :: Python :: 3",
        "Licence :: OSI Approved :: MIT Licence",
        "Operating System :: OS Independent"
    ],
    data_diles=[
        "data"
    ],
    packages=["incentive"],
    packages_dir={"incentive": "src/incentive"},
    python_requires=">-3.7",
)
