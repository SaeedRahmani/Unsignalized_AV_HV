from setuptools import setup, find_packages

setup(
    name="l5kit_conflict",
    version="0.1.0",
    packages=find_packages(where="l5kit_conflict"),
    package_dir={'': 'l5kit_conflict'},
    install_requires=[]
    ,
    author="Zhenlin (Gavin) Xu",
    author_email="gavinxu66@gmail.com",
    description="""
        A package to identify and analyse the traffic conflicts
        in the unsignalized intersection between AV and HV,
        using Woven by Toyota Prediction dataset (Lyft dataset).
    """,
    url="https://github.com/Zhenlin-Xu/unsignalized-intersection-conflicts-lyft",
    classifiers=[  # Classifiers help users find your project
        'Programming Language :: Python :: 3',
    ],
    python_requires="==3.8.18"
)