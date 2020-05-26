from distutils.core import setup
import glob

script_files = ['bin/simrep']+glob.glob('bin/*.py')

setup(
    name='SimRep',
    version='2.1',
    author='Wolfgang Kastaun',
    author_email='physik@fangwolg.de',
    packages=['simrep', 'simrep.plugins'],
    scripts=script_files,
    description='Automated postprocessing and report generation.',
    long_description=open('README.rst').read(),
    install_requires=[
        "PostCactus", "h5py",
        "numpy", "scipy", "matplotlib"
    ],
    python_requires='>=2.7, <3',
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Operating System :: OS Independent"
    ],
    package_data={'simrep':['data/*']}
)

