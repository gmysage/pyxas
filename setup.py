from setuptools import setup, find_packages

setup(
    name="pyxas",
    version="0.1.0",
    # find_packages() searches for folders with __init__.py
    packages=find_packages(),
    package_dir={"": "."}, 
    # List your dependencies here
    install_requires=[
        "pandas",
        "scikit-image",
        "scikit-learn",
        "numpy", 
        "matplotlib",
        "pystackreg",
        "xraylib",
        "h5py",

    ],
    entry_points={
        'console_scripts': [
            'run-pyxas = pyxas.pyxas_gui:main',
        ],
    },
    # Metadata
    author="Mingyuan Ge",
    description="A Python package for XAS data analysis",
    python_requires=">=3.7",
)

