from snippet_header import plt, np, finalize, GrainModel, evaluate_meanopacity

temp0   = 1e1
temp1   = 1e5
ntemp   = 1000
rhogas0 = 1e-10
rhogas  = np.ones(ntemp) * rhogas0
temp    = temp0 * (temp1 / temp0)**np.linspace(0., 1., ntemp)
dtg     = 0.01
abun    = [0.5, 0.5]
grain   = []
grain.append(GrainModel())
grain[0].load_standard_opacity('ddn01', 'silicate')
grain[0].sublimationmodel = ['tsubfrompeq', {'species': 'MgFeSiO4', 'plaw': -24}]

grain.append(GrainModel())
grain[1].load_standard_opacity('ddn01', 'waterice')
grain[1].sublimationmodel = ['tsubfrompeq', {'species': 'H2O', 'plaw': -10}]

rhodust = []
rhodust.append(rhogas * dtg * abun[0])
rhodust.append(rhogas * dtg * abun[1])

# plot

plt.figure()
mom = ['dustcomponents', {'method': 'simplemixing'}]
opc = evaluate_meanopacity(mom, rhogas, temp, rhodust=rhodust, grain=grain)
plt.plot(opc['temp'], opc['rosseland'], label='No sublimation')
mom = ['dustcomponents', {'method': 'simplemixing', 'autosublim': True}]
opc = evaluate_meanopacity(mom, rhogas, temp, rhodust=rhodust, grain=grain)
plt.plot(opc['temp'], opc['rosseland'], label='With sublimation')
mom = ['dustcomponents', {'method': 'simplemixing',
                          'autosublim': True, 'gasbelllin': True}]
opc = evaluate_meanopacity(mom, rhogas, temp, rhodust=rhodust, grain=grain)
plt.plot(opc['temp'], opc['rosseland'],
         label='With sublimation & gas', linewidth=2)
mom = ['belllin']
opc = evaluate_meanopacity(mom, rhogas, temp, rhodust=rhodust, grain=grain)
plt.plot(opc['temp'], opc['rosseland'], '--', label='Bell & Lin')
mom = ['belllin', {'onlygas': True}]
opc = evaluate_meanopacity(mom, rhogas, temp, rhodust=rhodust, grain=grain)
plt.plot(opc['temp'], opc['rosseland'], '--', label='Bell & Lin gas only')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('T [K]')
plt.ylabel(r'$\kappa_{\mathrm{R}} [\mathrm{cm}^2/\mathrm{g}]$')
plt.legend(loc='lower left')
plt.ylim(ymin=1e-7)
plt.title(r'Rosseland mean opacities with olivine + water ice + B&L gas')
plt.annotate('water snow line', xy=(160., 10), xycoords='data',
             xytext=(-60, +30), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2"))
plt.annotate('silicate sublimation', xy=(900., 3), xycoords='data',
             xytext=(-10, +50), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
# plt.annotate('molecules',xy=(2200., 2e-5), xycoords='data',
#             xytext=(-1, -50), textcoords='offset points',
#             arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=-.2"))
# plt.annotate('H-scattering',xy=(6000., 1e-2), xycoords='data',
#             xytext=(-1, -50), textcoords='offset points',
#             arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
# plt.annotate('free-free',xy=(13000., 1.5), xycoords='data',
#             xytext=(+10, +30), textcoords='offset points',
#             arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
# plt.annotate('electron scattering',xy=(40000., 0.2), xycoords='data',
#             xytext=(-70, -40), textcoords='offset points',
#             arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
# plt.savefig('fig_snippet_plot_dustopac_6_1.pdf')

finalize()
