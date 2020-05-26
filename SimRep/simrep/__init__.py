"""This package contains a framework to automatically generate a set of standard
plots from CACTUS simulations, and create HTML pages for convenient browsing.
The framework is plugin based. To add a new visualization module, one has to write
the scripts that create the plots (or other data) and a python module in which the 
HTML page contents are defined. The latter is based on a simple Python-based 
mini-language. The modules in this package are not intended to be used directly by
the end-user, but provide functionality for the simrep script. For plugin writers,
the modules plugins.repplugin needs to be included in the plugin's python module.
For visualization script writers, the module stdplot provides some standard 
functionality regarding matplotlib and data import."""

