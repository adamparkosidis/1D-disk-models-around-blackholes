from snippet_header import DiskRadialModel, plt, MS, au, LS, finalize

dtg           = 0.01
abun_silicate = 0.5
abun_waterice = 0.5

# setup

d = DiskRadialModel(mdisk=0.01 * MS, mstar=2 * MS, tstar=1e4, lstar=30 * LS, rin=0.03 * au)
d.add_dust(agrain=0.1e-4, xigrain=3.6, dtg=dtg * abun_silicate)
d.add_dust(agrain=0.1e-4, xigrain=1.0, dtg=dtg * abun_waterice)
d.dust[0].grain.load_standard_opacity('ddn01', 'silicate')
d.dust[1].grain.load_standard_opacity('ddn01', 'waterice')
d.dust[0].grain.sublimationmodel = ['tsubfrompeq', {'species': 'MgFeSiO4', 'plaw': -24}]
d.dust[1].grain.sublimationmodel = ['tsubfrompeq', {'species': 'H2O', 'plaw': -10}]
for dust in d.dust:
    assert dust.grain.agrain == dust.agrain.max()
    assert dust.grain.xigrain == dust.xigrain.max()

# plotting

plt.figure()
d.meanopacitymodel = ['dustcomponents', {'method': 'simplemixing'}]
d.compute_mean_opacity()
plt.plot(d.r / au, d.mean_opacity_rosseland, label='No sublimation')

d.meanopacitymodel = ['dustcomponents', {'method': 'simplemixing', 'autosublim': True}]
d.compute_mean_opacity()
plt.plot(d.r / au, d.mean_opacity_rosseland, label='With sublimation')

d.meanopacitymodel = ['dustcomponents', {'method': 'simplemixing', 'autosublim': True, 'gasbelllin': True}]
d.compute_mean_opacity()
plt.plot(d.r / au, d.mean_opacity_rosseland, label='With sublimation & gas')

d.meanopacitymodel = ['belllin']
d.compute_mean_opacity()
plt.plot(d.r / au, d.mean_opacity_rosseland, label='Bell & Lin')

d.meanopacitymodel = ['belllin', {'onlygas': True}]
d.compute_mean_opacity()
plt.plot(d.r / au, d.mean_opacity_rosseland, label='Bell & Lin gas only')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('r [au]')
plt.ylabel(r'$\kappa_{\mathrm{R}} [\mathrm{cm}^2/\mathrm{g}]$')
plt.legend(loc='lower right')
plt.ylim(bottom=1e-6)
plt.title('Rosseland mean opacities in the disk')
plt.savefig('fig_snippet_plot_dustopac_5_1.pdf')

finalize()
