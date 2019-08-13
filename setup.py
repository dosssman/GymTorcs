import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='gym_torcs',
    version='0.1',
    # scripts=['gym_torcs'],
    install_requires=['psutil', 'gym'],
    author="Rousslan F. J. Dossa",
    author_email="dosssman [at] hotmail.fr",
    description="A pip package for the Gym Torcs environment",
    long_description=long_description,
    # url=
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         # "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ]
)
