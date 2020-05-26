from distutils.core import setup

script_files = ['bin/simsync', 'bin/pardiff', 'bin/parprint']

setup(
    name='PostCactus',
    version='2.1',
    author='Wolfgang Kastaun',
    author_email='physik@fangwolg.de',
    packages=['postcactus'],
    scripts=script_files,
    license='LICENSE.txt',
    description='Read and postprocess CACTUS data',
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy", "scipy",
        "h5py",
        "matplotlib",
    ],
    python_requires='>=2.7, <3',
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Operating System :: OS Independent"
    ]
)
