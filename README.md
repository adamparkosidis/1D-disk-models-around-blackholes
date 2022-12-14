# 1D disk models around blackholes using DISKLAB code of Kees Dullemond and Til Birnstiel's

## Description

We produce numerical disks around $10M_{\odot}$ blackhole using DISKLAB and compare them with steady state analytical solutions. Conceptually, the starting point is the disk conservation equations. As first step, we only consider mass and angular momentum conservation: including the energy equation will also require us to keep track of changes in the disk scale-height and sound speed, which will entail self-consistent calculation of the kinematic viscosity as a function of radius at every time step. We first take a simpler approach, assuming that these changes are negligible (which would be the case if time variations in mass accretion through the disk are small). This way, we are be able to specify the radial dependence of the kinematic viscosity from the start. There are various ways to approach the problem, using changes of variable and a variety of formulations of the conservation equations. To enable a clearer understanding of the physical situation, we formulate them simply in terms of the mass accretion rate  ̇M, surface density Σ and kinematic viscosity $ν$. We implicitly assume that  ̇M and Σ are functions of radius and time, while ν is only a function of radius and is set by an $α$-viscosity prescription $ν = αcsH$ where $H$ is the (assumed fixed-in-time) scale height $H/R = const.$ and $α ∈ [0,1]$ the Shakura & Sunyaev viscosity parameter. Fortunately we don’t get into all the detailed mathematics because DISKLAB offers a neat implicit solver for the diffusion equation.
At the second part, we use your steady state disks to create thermal disk spectra directly from the local viscous heating rate. We make some simple assumptions about the emission from the accretion flow, to make some basic predictions about how the disk emission will vary as a function of photonenergy. By varying the mass accretion rate and inner disk radius (as a proxy for the black hole spin), we create template spectra that allow to infer these parameters from observations via the continuum fitting method.

## Description

There is a detailed analysis and graphs which capture the physics  in the `main.ipynb`, where the user can alter parameters of the system and further explore it behavior.

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

$$κ_{es} = 0.2 ×(1+X) \frac{cm2}{g}$$

and also a combined opacity law that takes both processes into account (a “mock version” by addition of the opacities will do fine here). In the above formulas, $X$ and $Z$ are hydrogen and “metal” abundances (’X’:0.7, ’Z’:0.02 are reasonable choices) and $g_{ff}$ should be set to one for our purposes. In `meanopacity.py` $ρ_g$ is called `rhogas`and $T$ is temperature We won’t need the Planck opacities, you can set them identical to the Rosseland mean.
