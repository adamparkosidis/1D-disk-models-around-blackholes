# 1D disk models of disks around black wholes using DISKLAB code of Kees Dullemond and Til Birnstiel's

## Introduction


We produce numerical disk calculation based on the differential equations
given below. For constant input  ̇M at the outer radius, you should be able to show that the
simulation converges on the expected steady state distribution of Σ(R). You can calculate the
steady state distribution using the appropriate equation for a standard steady disk (which relates ̇M to νΣ). Since you know  ̇M (it is the value put in at the outer radius) and ν(R) which results
from the chosen scale-height H/R = const., you can calculate Σ(R) and plot it against the radial
distribution obtained from running your code for long enough for the output to stabilise at the
steady state solution.
As a second step, you will incorporate different physical opacity prescriptions, allowing you
to obtain the scale-height H/R self-consistently from solving also the viscous heating for the disk
(the energy equation). Fortunately, disklab already has inbuilt functions for this task. You
should then compare your steady state disks with the analytic solutions given in the book of
Frank, King and Rayne.

We use your steady state disks to create thermal disk spectra directly from the local viscous heating rate. We make some simple assumptions about the emission from the accretion flow, to make some basic predictions about how the disk emission will vary as a function of photonenergy. By varying the mass accretion rate and inner disk radius (as a proxy for the black hole spin), we create template spectra that allow to infer these parameters from observations via the continuum fitting method.

## DISKLAB

## Installation and use

For notes on how to install and use `disklab`, please read the tutorial, which can be found in the `tutorial/` directory ([here](tutorial/disklab_tutorial.pdf)).

Typically you would install it by executing this in the top-level directory of this repository:

    pip install -e .

This installs a development version, i.e. if you modify the code, those changes will be reflected if you `import disklab`. To install a static version of the current code, leave out the `-e` option.

## Examples

All examples, including those from the [tutorial](tutorial/disklab_tutorial.pdf) can be found in the [snippets folder](snippets).

## Additions to the code

We implemented new opacity prescriptions into DISKLAB since it doesn’t have the relevant ones for accretion disks around compact objects (however it has a lot of dust physics which we will not need here). New opacities are be implemented in `meanopacity.py`. More specifically, we implement the ’kramer’ opacity law for free-free absorption and ’electron scattering’ opacity

$$κ_{ff} = 3.68 ×1022 g_{ff} (1 −Z)(1 + X)ρ_g T^{−7/2} \frac{cm2}{g}$$

$$κ_{ff} = 0.2 ×(1+X) \frac{cm2}{g}$$

and also a combined opacity law that takes both processes into account (a “mock version” by addition of the opacities will do fine here). In the above formulas, $X$ and $Z$ are hydrogen and “metal” abundances (’X’:0.7, ’Z’:0.02 are reasonable choices) and $g_{ff}$ should be set to one for our purposes. In `meanopacity.py` $ρ_g$ is called `rhogas`and $T$ is temperature We won’t need the Planck opacities, you can set them identical to the Rosseland mean.