import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="incentive",
    version="v1.1-alpha",
    author="Evripidis Gkanias",
    maintainer="Evripidis Gkanias",
    author_email="ev.gkanias@ed.ac.uk",
    maintainer_email="ev.gkanias@ed.ac.uk",
    description="A package implementing the incentive circuit in the fruit fly brain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/InsectRobotics/IncentiveCircuit",
    project_urls={
        "Bug Tracker": "https://github.com/InsectRobotics/IncentiveCircuit/issues"
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
    package_data={'incentive': [os.path.join('data', 'bennett2021', '*'),
                                os.path.join('data', 'handler2019', '*'),
                                os.path.join('data', 'fruitfly', '*'),
                                os.path.join('data', 'arena', '*'),
                                os.path.join('data', 'model-parameters.yml')]},
    python_requires=">=3.7",
)
