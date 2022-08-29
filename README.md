
# PostCactus

This repository contains a Python package named PostCactus for postprocessing 
data from numerical simulations performed with the 
[Einstein Toolkit](https://einsteintoolkit.org/).
The package provides a high-level Python interface to the various data formats 
in a simulation folder.

## Installation 

Currently, the onyl method is to clone the repo and pip-install locally from it.
We recommend to install into a [conda](https://docs.conda.io/en/latest/) Python evironment, even though there is no conda 
package for PostCactus yet.

```bash
git clone https://github.com/wokast/PyCactus.git
pip install ./pycactus/PostCactus
```

## Requirements
* Python >=3 (2.7 still supported but deprecated) 
* [h5py](https://www.h5py.org/) 
* [numpy, scipy](https://scipy.org/) 

Optional

* [jupyter](https://jupyter.org/) (recommended)
* [matplotlib](https://matplotlib.org/) (visualization)
* Sphinx (rebuild documentation)
* VTK (3D plots)


## Documentation
A copy of the docuentation can be found [here](https://wokast.github.io/PyCactus/)

Documentation can also be build from the repository. This requires Sphinx.

```bash
cd pycyctus/PostCactus/doc
make html
```

The folder `PostCactus/doc/examples/notebooks/` contains some example jupyter notebooks (note: not updated to Python3 yet)


# Other

Besides PostCactus, two commandline tools are included:

* `pardiff` compares parameter files.
* `simsync` uses rsync to transfer selected data types from simulation folders.
*  

This repository also contains two packages that are not maintained.
SimRep allows the automatic creation of html reports for a simulation, 
and the SimVideo package allows the creation of movies visualizing 
simulation data. For further information, see the README files of 
the individual packages.

## Similar packages

[Kuibit](https://sbozzolo.github.io/kuibit/) is a reverse-engineered rewrite of PostCactus, with improved packaging, user interface, and documentation. Performance may differ. The two interfaces are similar enough to try code with PostCactus in case Kuibit does not work.

[SimulationTools](https://simulationtools.org/index.html) is a free package for the non-free Mathematica environment, providing 
functionality very similar to PostCactus. 

