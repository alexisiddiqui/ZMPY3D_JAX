from setuptools import find_packages, setup

setup(
    name="ZMPY3D_JAX",
    version="0.0.2",
    author="Jhih Siang (Sean) Lai",
    author_email="js.lai@uqconnect.edu.au, jsl035@ucsd.edu",
    description="ZMPY3D JAX version",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tawssie/ZMPY3D",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    entry_points={
        "console_scripts": [
            "ZMPY3D_CLI_ZM=ZMPY3D_JAX.ZMPY3D_CLI_ZM:main",
            "ZMPY3D_CLI_SuperA2B=ZMPY3D_JAX.ZMPY3D_CLI_SuperA2B:main",
            "ZMPY3D_CLI_ShapeScore=ZMPY3D_JAX.ZMPY3D_CLI_ShapeScore:main",
            "ZMPY3D_CLI_BatchSuperA2B=ZMPY3D_JAX.ZMPY3D_CLI_BatchSuperA2B:main",
            "ZMPY3D_CLI_BatchShapeScore=ZMPY3D_JAX.ZMPY3D_CLI_BatchShapeScore:main",
            "ZMPY3D_CLI_BatchZM=ZMPY3D_JAX.ZMPY3D_CLI_BatchZM:main",
        ],
    },
    python_requires=">=3.9.16",
    install_requires=[
        "numpy>=1.23.5",
        "gdown",
        "ipykernel",
        "py3Dmol",
        "pytest",
    ],
    include_package_data=True,  # for the cache
)
