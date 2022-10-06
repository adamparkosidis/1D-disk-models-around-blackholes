#
# WHAT THIS SNIPPET_HEADER.PY IS MEANT FOR:
#
# This snippet_header.py file is only meant to make the snippet examples
# of DISKLAB a bit less cluttered with the usual import statements. But
# you can easily replace this with some (or all) of the statements below
# (with the exception of the stuff below the '----').
#
# For instance, the following three imports are the standard imports
# of most python scripts: os, numpy and plotting
#
import os
import numpy as np
import matplotlib.pyplot as plt
#
# And the following are imports from DISKLAB. Here they are all
# imported, but usually you need only a few of them, depending on
# which functionality of DISKLAB you want to use.
#
# For the 1-D radial models you need the DiskRadialModel class:
#
from disklab import DiskRadialModel
#
# If you want to add dust into the 1-D radial disk models, you need:
#
from disklab import DiskRadialComponent
#
# If you want to model 1-D vertical disk structure models, you need
# the DiskVerticalModel class:
#
from disklab import DiskVerticalModel
#
# And if you want to add dust into those, you need:
#
from disklab import DiskVerticalComponent
#
# If you want to do 2-D (radial-vertical) disk models, you need the
# Disk2D class:
#
from disklab import Disk2D
#
# If you want to include details of the dust grain physics, you need:
#
from disklab import GrainModel
#
# If your disk model needs information about mean opacities, you need:
#
from disklab.meanopacity import evaluate_meanopacity
#
# And for your convenience, here are some natural constants (year,
# astronomical unit, solar mass, earth mass, Boltzmann constant,
# proton mass and solar luminosity):
#
from disklab.natconst import year, au, MS, Mea, kk, mp, LS, GG
#
# If you wish to make convenient wireframe surface plots:
#
from snippet_plottingtools import surface
#
# this one is used to read command line parameters for executing the snippets
# non-interactively.
#
import argparse

# for backward compatibility with matplotlib<2.0.0, we define our own color list
# to be used instead of a color_cycler, this should work like a periodic list


class Colors(list):
    colorlist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                 '#bcbd22', '#17becf']

    def __init__(self, colorlist=colorlist):
        self.colorlist = colorlist

    def __getitem__(self, i):
        return self.colorlist[i % len(self.colorlist)]


colors = Colors()

# define which parts should be imported upon `from snippet_header import *`

__all__ = ['np',
           'plt',
           'DiskRadialModel',
           'is_interactive',
           'year',
           'au',
           'MS',
           'Mea',
           'kk',
           'mp',
           'LS',
           'GG',
           'DiskRadialComponent',
           'Disk2D',
           'DiskVerticalModel',
           'DiskVerticalComponent',
           'GrainModel',
           'evaluate_meanopacity',
           'colors',
           'surface'
           ]

#
# ---------------------------------------------------------------
# THE STUFF BELOW IS ONLY FOR THE AUTOMATIC RUNNING OF ALL
# SNIPPETS FOR THE AUTOMATIC TUTORIAL GENERATION.
#
# When you use or modify the snippets yourself, you can replace the
# command finalize() by plt.show(). You do not really need the
# finalize() command.
# ---------------------------------------------------------------
#
RTHF   = argparse.RawTextHelpFormatter
PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=RTHF)
PARSER.add_argument('-n', '--nonstop', help='do not show the plots, instead proceed', action='store_true', default=False)
PARSER.add_argument('--hash', help='create a hash of arrays passed to finalize() via the keyword result, and write to file result.hash', action='store_true', default=False)
PARSER.add_argument('-t', '--test', help='compare the hash of arrays passed to finalize() to file result.hash', action='store_true', default=False)
PARSER.add_argument('--matplotlib', action='store_true', default=False)
ARGS  = PARSER.parse_args()


def is_interactive():
    """
    Returns true if script is run interactively, or
    if
    """
    import __main__ as main

    return not hasattr(main, '__file__')


def finalize(figures=None, results=None):
    """
    Run this at the end of every snippet:

    IF
        script is run interactively
    OR
        script is run from command line with option `-i`
    THEN:
        plt.show()

    ELSE:
        save all figures (or the ones that are passed to this function)
        if no figures should be saved: pass an empty list.

    Keywords:
    ---------

    figures : None | list
        list: all figure handles to be saved, none are saved if list is empty
        None: saves all figures

    """
    from __main__ import __file__

    # get all figure numbers (or the ones given as argument)

    if figures is None:
        fignums = plt.get_fignums()
    elif figures is []:
        return 0
    else:
        fignums = [f.number for f in figures]

    # either show the figures, or save them as PDFs, unless testing mode

    if not ARGS.hash and not ARGS.test:
        if not ARGS.nonstop:
            plt.show()
        else:
            for i in fignums:
                fname = 'fig_' + os.path.basename(__file__).replace('.py', '_{}.pdf'.format(i))
                print('saving {}'.format(fname))
                plt.figure(i).savefig(fname)

    # If requested, generate hash of results and write to file

    if ARGS.hash and results is not None:
        from disklab.utilities import hash_arrays
        hashhex = hash_arrays(results)
        fname = 'results_' + os.path.basename(__file__).replace('.py', '.hash')
        with open(fname, 'w') as f:
            f.write(hashhex)

    # If requested, generate hash and compare to hash from file

    if ARGS.test and results is not None:
        from disklab.utilities import hash_arrays
        hashhex = hash_arrays(results)
        fname = 'results_' + os.path.basename(__file__).replace('.py', '.hash')
        with open(fname, 'r') as f:
            hashold = f.readline()
        if hashhex == hashold:
            print('Testing ' + os.path.basename(__file__) + ' was successful.')
        else:
            raise ValueError('Testing ' + os.path.basename(__file__) + ' FAILED.')
