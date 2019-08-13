import setuptools

# with open("README.md", "r") as fh:
    # long_description = fh.read()
long_description = "Please check out the README at: https://github.com/dosssman/GymTorcs"

setuptools.setup(
    name='gym_torcs',
    version='0.1.1',
    # scripts=['gym_torcs'],
    install_requires=['psutil', 'gym'],
    author="Rousslan F. J. Dossa",
    author_email="dosssman@hotmail.fr",
    description="A pip package for the Gym Torcs environment",
    long_description=long_description,
    url="https://github.com/dosssman/GymTorcs.git",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         # "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ]
)
