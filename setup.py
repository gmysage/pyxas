import setuptools

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

requirements = ['setuptools'] + requirements

setuptools.setup(
     name='pyxas',  
     version='1.0',
     scripts=['pyxas_gui'] ,
     author="Mingyuan Ge",
     author_email="gmysage@gmail.com",
     description="A python package for 2D/3D xanes analysis",
     url="https://github.com/gmysage/pyxas",
     packages=setuptools.find_packages(),
     install_requires=requirements,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.6',
 )
