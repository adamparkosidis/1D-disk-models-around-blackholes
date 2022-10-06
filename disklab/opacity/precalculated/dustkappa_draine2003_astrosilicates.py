from disklab import opacity
import numpy as np

# define optical constants

diel_const = opacity.diel_draine2003_astrosil()

# define wave length and particle size grids and material density

lam   = np.logspace(-6, 0, 200)
a     = np.logspace(-5, 0, 150)
rho_s = 3.5

# call the Mie calculation

opac_dict = opacity.get_opacities(a, lam, rho_s=rho_s, diel_const=diel_const, extrapolate_large_grains=False)

# store the opacity in 'myopac.npz'

opacity.write_disklab_opacity('dustkappa_draine2003_astrosilicates', opac_dict)
