from snippet_header import np, plt, finalize
from disklab.meanopacity import belllin

rho=np.ones(100)*1e-10              # Gas density [g/cm^3]
temp=10*1e4**np.linspace(0,1,100)   # Grid of temperatures [K]

plt.figure()
kap = belllin(rho,temp)
plt.loglog(temp,kap,label='Normal dust content')
dustfactor=1e-2
kap=belllin(rho,temp,dustfactor=dustfactor)
plt.loglog(temp,kap,label='Dust reduced by 100')
plt.xlabel('T [K]')
plt.ylabel(r'$\kappa_{\mathrm{Ross}}\; [\mathrm{cm}^2/\mathrm{g}]$')
plt.legend()

finalize()


