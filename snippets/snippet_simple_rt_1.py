from snippet_header import DiskRadialModel, plt, au, finalize

d1 = DiskRadialModel(flang=0.05)
flang = 0.05 * (d1.r / au)**(2. / 7.)
d2 = DiskRadialModel(flang=flang)
plt.plot(d1.r / au, d1.tmid, label='flang=0.05')
plt.plot(d2.r / au, d2.tmid, label='flang=0.05*(r/au)^(2/7)')
plt.legend()
plt.xlabel(r'$r [\mathrm{au}]$')
plt.ylabel(r'$T [\mathrm{K}]$')
plt.xscale('log')
plt.yscale('log')

finalize(results=(d1.tmid,d2.tmid))
