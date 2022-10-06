from snippet_header import DiskVerticalModel, np, plt, MS, LS, au, finalize
import copy

mstar  = 1 * MS
lstar  = 1 * LS
r      = 1 * au
siggas = 1700.
kapdst = 1e2
flang  = 0.05
zrmax  = 0.2
alpha  = 1e-4
agrain = 1e-1
dtg    = 0.5    # Yes, this is a huge dust-to-gas ratio!
opac = ['supersimple', {'dusttogas': 0.01, 'kappadust': 1e2}]  # Only for opacity!
vert = DiskVerticalModel(mstar, r, siggas, flang=flang, zrmax=zrmax,
                         lstar=lstar, meanopacitymodel=opac,
                         alphavisc=alpha,nz=200)
vert.iterate_vertical_structure()
vert.add_dust(agrain,dtg=dtg)
vert.dust[0].compute_settling_mixing_equilibrium()
vertorig = copy.deepcopy(vert)
    
# Now include the dust loading
for iter in range(3):
    vert.iterate_vertical_structure(dtgitermax=20,dtgerrtol=0.02)
    vert.dust[0].compute_settling_mixing_equilibrium()

# Check dust mass conservation
vertorig.dust[0].compute_surfacedensity()
vert.dust[0].compute_surfacedensity()
assert np.abs(vert.dust[0].sigma/vertorig.dust[0].sigma-1.)<1e-3

# plotting
plt.figure()
plt.plot(vert.z / vert.r, vertorig.rhogas,label='Gas Orig')
plt.plot(vert.z / vert.r, vert.rhogas,label='Gas Dust-laden')
plt.gca().set_prop_cycle(None)
plt.plot(vert.z / vert.r, vertorig.dust[0].rho,':',label='Dust Orig')
plt.plot(vert.z / vert.r, vert.dust[0].rho,':',label='Dust Dust-laden')
plt.xlabel('z/r')
plt.ylabel(r'$\rho [\mathrm{g}/\mathrm{cm}^3]$')
plt.xlim(left=0,right=0.1)
plt.legend()

finalize()
