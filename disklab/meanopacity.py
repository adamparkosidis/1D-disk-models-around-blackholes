"""
This file contains a set of subroutines for computing mean opacities
(Planck and Rosseland means). It uses the grainmodel.py objects, but
is otherwise independent of the disklab modules.

The main inputs are:

   meanopacitymodel            A list which defines the mean opacity
                               model to use
   rhogas                      The gas density
   temp                        The temperature
"""
import numpy as np
from .path_interpol_2d import path_interpol_2d
from .grainmodel import mixedgrains


def evaluate_meanopacity(meanopacitymodel, rhogas, temp, rhodust=None, grain=None):
    """
    The main function returning the values of the mean opacity for
    an array of rhogas and temp values.

    ARGUMENTS:
       meanopacitymodel          A list. Zeroth element is a string which
                                 specifies which mean opacity model to use.
                                 The next elements contain further information
                                 needed for that opacity model.
       rhogas                    Gas density in g/cm^3. Can be an array.
       temp                      Temperature in K. Can be an array.
       rhodust                   List of dust densities in g/cm^3. Can be a list of arrays.
                                 For one dust species the list contains only 1 array; for
                                 two dust species, it is a list of 2 arrays.
       grain                     List of grain models. Not an array! Only a list of
                                 grainmodel objects. See grainmodel.py

    RETURNS:
       Dictionary of mean opacities:
            ['kappa_planck']     The Planck mean opacity in cm^2 per gram of gas.
            ['kappa_rosseland']  The Rosseland mean opacity in cm^2 per gram of gas.
       NOTE: These opacities are per gram of *gas*. So to obtain the extinction
             coefficient you must multiply by rhogas.

    OPACITY MODELS AVAILABLE:

          'supersimple'     The simplest opacity model available: you simply specify
                            a value to be used. The second element of the meanopacitymodel
                            list is a dictionary with information about this opacity.
                            The simplest way is:

                             meanopacitymodel = ['supersimple',{'kappagas':1e0}]

                            You can also specify a hypothetical dust opacity, and a dust-to-gas
                            ratio:

                             meanopacitymodel = ['supersimple',{'dusttogas':0.01,'kappadust':1e2}]

                            (note that both examples give the same result).

                            You can also use this model to specify a z-dependent opacity,
                            simply by giving an array instead of a value:

                             kappaarray       = np.ones_like(self.rhogas)
                             meanopacitymodel = ['supersimple',{'kappagas':kappaarray}]

                            where of course you set kappaarray to some more useful
                            array than just np.ones_like(self.rhogas), as long as it
                            has the same number of elements as self.rhogas.

          'dustcomponents'  This will construct the mean opacity arrays from the available
                            dust components in the disk (or from the dust components you
                            explicitly give as arguments to this method). The second
                            element of the list is a dictionary with further settings.
                            The most important dictionary element is 'method'. It can
                            have the following values:

                            method = 'fullcalculation':
                                      Compute the mean opacities from the full
                                      frequency-dependent opacities of the dust components.
                                      This model is very time-consuming, because it will
                                      recalculate the integrals over the full frequency grids.
                                      For fast calculations of disk models, this is not ideal.
                                      But there can be circumstances by which this is the
                                      better method: if different dust species mix/drift
                                      in different ways, so that the composition of the
                                      dust mixture is different at different times.

                            method = 'simplemixing':
                                      Use mean opacities from each of the dust species
                                      individually, and average them weighted by abundance.
                                      For a single dust species this is of course exact.
                                      For multiple dust species this is not quite correct,
                                      but it is fast. The reason why it is not quite
                                      correct is that the Rosseland mean of the mixed
                                      frequency dependent opacity is not equal to the
                                      average of the Rosseland mean of the individual
                                      dust opacities. It is only correct if the frequency-
                                      dependent opacities are 'correlated'. In atmospheric
                                      radiative transfer physics they call this the
                                      'correlated k assumption'. It is the method of
                                      choice if one wants to allow time- and space-
                                      varying dust abundances, while avoiding the time-
                                      consuming 'fullcalculation' method.

                            So if you choose 'simplemixing' then this if what the meanopacitymodel
                            should look like:

                             meanopacitymodel = ['dustcomponents',{'method':'simplemixing'}]

                            The sublimation of the grains can either be handled by yourself,
                            by simply reducing the density of the dust in the regions where
                            this dust sublimated. Or you can ask the mean opacity model to
                            do this automatically (where the dust density thus takes the
                            meaning of 'combined dust and vapor density'). You can turn on
                            this automatic sublimation by adding 'autosublim':True, as in:

                             meanopacitymodel = ['dustcomponents',{'method':'simplemixing','autosublim':True}]


          'tabulated'       This method will look up the opacity from a table that you
                            provide, and it will interpolate in that table. The meanopacitymodel
                            has to contain this table:

                             meanopacitymodel = ['tabulated',{'rhogrid':rhogas,'tempgrid':temp, \
                                                              'kappa_planck':kappa_planck,      \
                                                              'kappa_rosseland':kappa_rosseland,\
                                                              'method':'linear'}]

                            where rhogas[0:nrho] and temp[0:ntemp] are two 1-D arrays giving the
                            coordinates of the table, and kappa_planck[0:nrho,0:ntemp] and
                            kappa_rosseland[0:nrho,0:ntemp] are the 2-D tables of opacity in
                            units of cm^2/gram-of-gas.

                            Note that since these opacities can be extremely steep functions of
                            density and temperature, it is often better to do the tabulation in
                            logarithmic form. You can do this, by specifying in the meanopacitymodel
                            instead of 'kappa_planck' and 'kappa_rosseland' the arrays
                            'ln_kappa_planck' and 'ln_kappa_rosseland', which then should be the
                            natural logarithms of kappa_planck and kappa_rosseland, respectively:

                             ln_kappa_planck    = np.log(kappa_planck)
                             ln_kappa_rosseland = np.log(kappa_rosseland)
                             meanopacitymodel = ['tabulated',{'rhogrid':rhogas,'tempgrid':temp,       \
                                                              'ln_kappa_planck':ln_kappa_planck,      \
                                                              'ln_kappa_rosseland':ln_kappa_rosseland,\
                                                              'method':'linear'}]

                            The resulting interpolated values will then automatically be
                            np.exp(...) again, so that the result is again the actual opacity instead
                            of the np.log(...) of the opacity.

          'belllin'         This is the well-known Bell & Lin (1994) ApJ 427, 987 opacity table.
                            Set meanopacitymodel to:

                             meanopacitymodel = ['belllin']

    """
    #
    # If rhogas and temp are scalar, make array
    #
    if np.isscalar(rhogas):
        rhogas = [rhogas]
    if temp is not None:
        if np.isscalar(temp):
            temp   = [temp]
        #
        # Check lengths
        #
        assert len(rhogas) == len(temp), 'Rhogas and temp must have same dimension'
    #
    # Make the output arrays
    #
    mean_opacity_planck    = np.zeros_like(rhogas)
    mean_opacity_rosseland = np.zeros_like(rhogas)
    #
    # If rhodust and/or grain is set, they must be lists (one element for
    # each dust species)
    #
    if rhodust is not None:
        if type(rhodust) != list:
            rhodust = [rhodust]
    if grain is not None:
        if type(grain) != list:
            grain = [grain]
    #
    # Now handle the different mean opacity models
    #
    if len(meanopacitymodel) > 1:
        params = meanopacitymodel[1]
    else:
        params = None
    if meanopacitymodel[0] == 'supersimple':
        #
        # Simple constant mean opacity
        #
        assert len(meanopacitymodel) > 1, 'Supersimple opacity model requires at least the opacity value'
        if type(meanopacitymodel[1]) == dict:
            kappa = None
            if type(meanopacitymodel[1]) == float:
                kappa = meanopacitymodel[1]
            else:
                if 'kappagas' in meanopacitymodel[1]:
                    kappa = meanopacitymodel[1]['kappagas']
                if 'kappadust' in meanopacitymodel[1]:
                    assert 'dusttogas' in meanopacitymodel[1], 'If you specify kappadust you must also specify dusttogas.'
                    kappa = meanopacitymodel[1]['kappadust'] * meanopacitymodel[1]['dusttogas']
            assert kappa is not None, 'Error in supersimple opacity: could not evaluate opacity.'
            mean_opacity_planck[:] = kappa
            mean_opacity_rosseland[:] = kappa
    elif meanopacitymodel[0] == 'dustcomponents':
        #
        # Use the dust species to compute the mean opacity.
        #
        # Check if they are available
        #
        assert rhodust is not None, "Sorry, we cannot compute a mean opacity from the dust components if we do not have the dust densities or abundances specified."
        assert grain is not None, "Sorry, we cannot compute a mean opacity from the dust components if we do not have the grain models specified."
        #
        # They should have the same length
        #
        assert len(rhodust) == len(grain), "Sorry, the number of species of rhodust is not equal to the number of grain models."
        nspec = len(rhodust)
        #
        # Now check which method to use
        #
        assert 'method' in params, "For mean opacity model dustcomponents, must specify method."
        if 'autosublim' in params:
            autosublim = params['autosublim']
        else:
            autosublim = False
        if params['method'] == 'fullcalculation':
            #
            # The full re-computation of the mean opacities
            #
            mix = mixedgrains(grain)    # CAREFUL: MAYBE SET WAVELENGTHGRID
            nn  = len(rhogas)
            for ii in range(nn):
                abundancelist = []
                for ispec in range(nspec):
                    abundancelist.append(rhodust[ispec][ii] / rhogas[ii])
                mix.compute_mixed_opacity(abundancelist, temp=temp[ii], trad=None,
                                          rhogas=rhogas[ii], autosublim=autosublim)   # Must add trad later
                mean_opacity_planck[ii]    = mix.kappa_planck
                mean_opacity_rosseland[ii] = mix.kappa_rosseland
        elif params['method'] == 'simplemixing':
            #
            # The simple averaging of the mean opacities of the dust species
            #
            mean_opacity_planck[:]    = 0.0
            mean_opacity_rosseland[:] = 0.0
            for ispec in range(nspec):
                abun = rhodust[ispec] / rhogas
                if autosublim:
                    if grain[ispec].sublimationmodel is not None:
                        abun = grain[ispec].abundance_after_sublimation(abun, rhogas, temp)
                mean_opacity_planck[:]    += abun[:] * grain[ispec].get_planckmean_tabulated_without_sublimation(temp[:])
                mean_opacity_rosseland[:] += abun[:] * grain[ispec].get_rosselandmean_tabulated_without_sublimation(temp[:])
        else:
            raise ValueError('Method for mean opacity using dust components not known.')
        #
        # If requested, add the gas-only part of the Bell & Lin (1994) ApJ 427, 987 opacity
        #
        if 'gasbelllin' in params:
            if params['gasbelllin']:
                kappa = belllin(rhogas, temp, onlygas=True)
                mean_opacity_planck[:]    += kappa[:]
                mean_opacity_rosseland[:] += kappa[:]
    elif meanopacitymodel[0] == 'tabulated':
        #
        # A tabulated opacity
        #
        assert 'tempgrid' in params, "Tabulated mean opacities requires temperature grid tgrid"
        assert 'rhogrid' in params, "Tabulated mean opacities requires density grid rhogrid"
        assert (('kappa_planck' in params and 'kappa_rosseland' in params) or
                ('ln_kappa_planck' in params and 'ln_kappa_rosseland' in params)), \
            "Tabulated mean opacities requires tabulated kappa_planck and kappa_rosseland (either in linear or in log form)"
        method = 'linear'
        if 'method' in params:
            method = params['method']
        if 'kappa_planck' in params:
            mean_opacity_planck[:]    = path_interpol_2d(params['kappa_planck'], params['rhogrid'], params['tempgrid'], rhogas, temp, method=method, use_interp2d=False)
            mean_opacity_rosseland[:] = path_interpol_2d(params['kappa_rosseland'], params['rhogrid'], params['tempgrid'], rhogas, temp, method=method, use_interp2d=False)
        else:
            ln_rhogrid  = np.log(params['rhogrid'])
            ln_tempgrid = np.log(params['tempgrid'])
            ln_rhogas   = np.log(rhogas)
            ln_temp     = np.log(temp)
            mean_opacity_planck[:]    = np.exp(path_interpol_2d(params['ln_kappa_planck'], ln_rhogrid, ln_tempgrid, ln_rhogas, ln_temp, method=method, use_interp2d=False))
            mean_opacity_rosseland[:] = np.exp(path_interpol_2d(params['ln_kappa_rosseland'], ln_rhogrid, ln_tempgrid, ln_rhogas, ln_temp, method=method, use_interp2d=False))
    elif meanopacitymodel[0] == 'belllin':
        #
        # The Bell & Lin (1994) ApJ 427, 987 opacity model
        #
        onlygas    = False
        dustfactor = 1.0
        if params is not None:
            if 'onlygas' in params:
                onlygas = params['onlygas']
            if 'dustfactor' in params:
                dustfactor = params['dustfactor']
        kappa = belllin(rhogas, temp, onlygas=onlygas,dustfactor=dustfactor)
        mean_opacity_planck[:] = kappa[:]
        mean_opacity_rosseland[:] = kappa[:]
    elif meanopacitymodel[0] == 'kramer':                                     # <---- I did that!
        g_ff= 1.0
        X = 0.7
        Z = 0.02
        kappa = 3.68e22*g_ff*(1.0-Z)*(1.0+X)*rhogas*(temp**(-7.0/2.0))
        mean_opacity_planck[:] = kappa[:]
        mean_opacity_rosseland[:] = kappa[:]
    elif meanopacitymodel[0] == 'electron':                                     # <---- I did that!
        X = 0.7
        kappa = 0.2*(1.0+X) + 0.0*temp
        mean_opacity_planck[:] = kappa[:]
        mean_opacity_rosseland[:] = kappa[:]
    elif meanopacitymodel[0] == 'kramer_electron':                                     # <---- I did that!
        g_ff= 1.0
        X = 0.7
        Z = 0.02
        kappa = 3.68e22*g_ff*(1.0-Z)*(1.0+X)*rhogas*(temp**(-7.0/2.0)) + 0.2*(1.0+X)
        mean_opacity_planck[:] = kappa[:]
        mean_opacity_rosseland[:] = kappa[:]
    else:
        #opedit:
        #
        # Add your custom opacity law here!
        #
        raise ValueError('Mean opacity model not known.')
    #
    # Return
    #
    return {'rhogas': rhogas, 'temp': temp, 'planck': mean_opacity_planck, 'rosseland': mean_opacity_rosseland}


def belllin(rho, temp, onlygas=False,dustfactor=1.0):
    """
    The Bell & Lin (1994) ApJ 427, 987 mean opacities.

    ARGUMENTS:
     rho           Gas density in g/cm^3
     temp          Temperature in K
     onlygas       If set, then do not include the dust part of the opacity
     dustfactor    To mimick dust loss (e.g. conversion of dust into planets)
                   set dustfactor to a number smaller than 1.0
    """
    def func(ki,a,b,rho,temp,special=False):
        n   = len(rho)
        nn  = len(ki)
        dm  = np.zeros((nn, n))
        for ii in range(nn):
            dm[ii, :] = ki[ii] * rho[:]**a[ii]
        tc  = np.zeros((nn + 1, n))
        for ii in range(nn - 1):
            tc[ii + 1, :] = (dm[ii, :] / dm[ii + 1, :])**(1. / (b[ii + 1] - b[ii]))
        tc[nn, :] = 1e99
        if special:
            tcspecial = (dm[nn-3, :] / dm[nn-1, :])**(1. / (b[nn-1] - b[nn-3]))
            mask = tc[nn-1,:]<tcspecial
            tc[nn-2,mask] = tcspecial[mask] # Avoid unphysical jump at low rho
            tc[nn-1,mask] = tcspecial[mask] # Avoid unphysical jump at low rho
        kappa = np.zeros_like(rho)
        for ii in range(nn):
            idx = np.where(np.logical_and(temp > tc[ii, :], temp < tc[ii + 1, :]))
            kappa[idx] = dm[ii, idx] * temp[idx]**b[ii]
        return kappa
    gas_ki   = np.array([1e-8, 1e-36, 1.5e20, 0.348])
    gas_a    = np.array([0.6667, 0.3333, 1., 0.])
    gas_b    = np.array([3., 10., -2.5, 0.])
    dust_ki  = np.array([2e-4, 2e16, 0.1, 2e81])
    dust_a   = np.array([0., 0., 0., 1.])
    dust_b   = np.array([2., -7., 0.5, -24.])
    dust_ki *= dustfactor
    scl = False
    if np.isscalar(rho):
        rho = np.array([rho])
        scl = True
    if np.isscalar(temp):
        temp = np.array([temp])
    assert len(rho) == len(temp), "Bell and Lin opacity: Length of array of rho and temp must be equal"
    kappa    = func(gas_ki,gas_a,gas_b,rho,temp,special=True)
    if not onlygas:
        kappa += func(dust_ki,dust_a,dust_b,rho,temp)
    if scl:
        kappa = kappa[0]
    return kappa
