from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
    this function will return the list of requirements
    """
    requirements=[]
    with open(file_path) as requirements_obj:
        requirements=requirements_obj.readlines()
        requirements=[req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="multimodal_bitcoin_price_prediction",
    version="0.1",
    description="A project to use Bitcoin price charts and onchain metrics to predict price direction for the next day",
    author="Oluwadamilare Omole",
    author_email="oluwadamilare.omole@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    install_requires=get_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "bitcoin_price_prediction=src.main:main",
        ],
    },
    
    #url="https://github.com/Stevenomole/Multimodal_bitcoin_price_prediction.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)