#
# Interactive plotting tool
#
# Copyright (c) 2018 C.P. Dullemond
# Free software under the standard MIT License
#
import numpy as np

def interactive_plot(x, func, params, ymin=None, ymax=None, parnames=None, parunits=None, fig=None, ax=None, axmodel=None, parstart=None, iparstart=None, plotbutton=False, fixedpar=None, returnipar=False, block=False, paramsalt=None, altformat='', **kwargs):
    """
    Plot the function func(x) with parameters given by the params
    list of lists. 

    ARGUMENTS:
      x          Array of x values
      func       Function func(x,params)
      params     List of parameters, but with each parameter value
                 here given as a list of possible values.

    OPTIONAL ARGUMENTS:
      ymin       Set vertical axis lower limit
      ymax       Set vertical axis upper limit
      parnames   Names of the params, e.g. ['A', 'omega']
                 If the parnames have an '=' sign (e.g. ['A = ', 'omega = '])
                 then the value of the parameters are written out.
      parunits   If set, a list of values by which the parameter values are divided
                 before being printed on the widget (only if parnames have '=').
                 It only affects the printing next to the sliders, and has no 
                 other effect.
      fig        A pre-existing figure
      ax         A pre-existing axis
      axmodel    If set, this is the plot style of the model
      parstart   If set, set the sliders initially close to these values
      iparstart  If set, set the slider index values initially to these values
                 (note: iparstart is an alternative to parstart)
      paramsalt  If set, then instead of the params values, the paramsalt values 
                 will be written after '=' (only if parnames is set, see above).
      returnipar If True, then return ipar
      block      If True, then wait until window is closed

    EXAMPLE 1 (Simplest example):
    from interactive_plot import *
    def func(x,param): return param[0]*np.sin(param[1]*x)
    x      = np.linspace(0,2*np.pi,100)
    params = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)] # Choices of parameter values
    interactive_plot(x, func, params, ymax=1., ymin=-1., parnames=['A = ','omega = '])

    EXAMPLE 1-a (With plotting button instead of automatic replot; useful for heavier models):
    from interactive_plot import *
    def func(x,param): return param[0]*np.sin(param[1]*x)
    x      = np.linspace(0,2*np.pi,100)
    params = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)] # Choices of parameter values
    interactive_plot(x, func, params, ymax=1., ymin=-1., parnames=['A = ','omega = '],plotbutton=True)

    EXAMPLE 1-b (Plotting the content of a pre-calculated 2-D array)
    from interactive_plot import *
    x       = np.linspace(0,2*np.pi,100)
    y_array = np.zeros((30,100))
    omega   = np.linspace(1,3.,30)
    for i in range(30): y_array[i,:] = np.sin(omega[i]*x)
    def func(x,param): return y_array[param[0],:]
    params  = [np.arange(30)] # Choices of parameter values
    interactive_plot(x, func, params)

    EXAMPLE 2 (Model fitting to data):
    import numpy as np
    import matplotlib.pyplot as plt
    from interactive_plot import *
    def func(x,param): return param[0]*np.sin(param[1]*x)
    x        = np.linspace(0,2*np.pi,100)
    data     = 0.5*np.sin(2.*x)*(1.0+0.6*np.random.normal(size=len(x)))
    fig      = plt.figure(1)
    ax       = plt.axes(xlim=(x.min(),x.max()),ylim=(-1.2,1.2))
    axd,     = ax.plot(x,data,'o',label='data')
    plt.xlabel('x [cm]')
    plt.ylabel('f [erg/s]')
    params   = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)] # Choices of parameter values
    parstart = [0.6,2.0]  # Initial guesses for parameters
    interactive_plot(x, func, params, parnames=['A = ','omega = '], fig=fig, ax=ax, label='model',parstart=parstart)
    ax.legend()
    plt.show()

    EXAMPLE 2-a (Model overplotting over an image):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from interactive_plot import *
    def func(x,param): return param[0]*np.sin(param[1]*x)
    x        = np.linspace(0,2*np.pi,100)
    image    = np.random.normal(size=(70,70)) # Make some image
    fig      = plt.figure(1)
    extent   = [x.min(),x.max(),-1.2,1.2]
    axd      = plt.imshow(image,extent=extent,cmap=cm.hot)
    ax       = plt.gca()
    plt.axis(extent)
    plt.xlabel('x [cm]')
    plt.ylabel('f [erg/s]')
    params   = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)] # Choices of parameter values
    parstart = [0.6,2.0]  # Initial guesses for parameters
    interactive_plot(x, func, params, parnames=['A = ','omega = '], fig=fig, ax=ax, label='model',parstart=parstart)
    ax.legend()
    plt.show()

    EXAMPLE 3 (Fitting two models simultaneously to data):
    import numpy as np
    import matplotlib.pyplot as plt
    from interactive_plot import *
    def func(x,param): return np.vstack((param[0]*np.sin(param[1]*x),param[0]*np.cos(param[1]*x)))
    x      = np.linspace(0,2*np.pi,100)
    data   = 0.5*np.sin(2.*x)*(1.0+0.6*np.random.normal(size=len(x)))
    fig    = plt.figure(1)
    ax     = plt.axes(xlim=(x.min(),x.max()),ylim=(-1.2,1.2))
    axd,   = ax.plot(x,data,'o',label='data')
    axm0,  = ax.plot(x,data,'--',label='sin')
    axm1,  = ax.plot(x,data,':',label='cos')
    axmodel= [axm0,axm1]
    plt.xlabel('x [cm]')
    plt.ylabel('f [erg/s]')
    params = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)]
    interactive_plot(x, func, params, parnames=['A = ','omega = '], fig=fig, ax=ax, axmodel=axmodel)
    ax.legend()
    plt.show()

    EXAMPLE 3-a (Fitting two models in two separate plots simultaneously):
    import numpy as np
    import matplotlib.pyplot as plt
    from interactive_plot import *
    def func(x,param): return np.vstack((param[0]*np.sin(param[1]*x),param[0]*np.cos(param[1]*x)))
    x         = np.linspace(0,2*np.pi,100)
    data      = 0.5*np.sin(2.*x)*(1.0+0.6*np.random.normal(size=len(x)))
    extent    = [x.min(),x.max(),-1.2,1.2]
    fig, axes = plt.subplots(ncols=2)
    axes[0].axis(extent)
    axes[1].axis(extent)
    axd0,  = axes[0].plot(x,data,'o',label='data')
    axm0,  = axes[0].plot(x,data,'--',label='sin')
    axd1,  = axes[1].plot(x,data,'o',label='data')
    axm1,  = axes[1].plot(x,data,':',label='cos')
    axmodel= [axm0,axm1]
    params = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)]
    interactive_plot(x, func, params, parnames=['A = ','omega = '], fig=fig, ax=0, axmodel=axmodel)
    plt.show()

    EXAMPLE 4: (passing additional fixed parameters to function):
    from interactive_plot import *
    def func(x,param,fixedpar={}): return param[0]*np.sin(param[1]*x)+fixedpar['offset']
    x      = np.linspace(0,2*np.pi,100)
    params = [np.linspace(0.1,1.,30),np.linspace(1.,3.,30)] # Choices of parameter values
    interactive_plot(x, func, params, ymax=1., ymin=-1., parnames=['A = ','omega = '],fixedpar={'offset':0.6})
    
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons

    # Compute spacing of plot, sliders and button
    hslider  = 0.03
    nslidrscl= 6
    if(len(params)>nslidrscl):
        hslider *= float(nslidrscl)/len(params)
    dyslider = hslider*(4./3.)
    xslider  = 0.3
    wslider  = 0.3
    hbutton  = 0.06
    wbutton  = 0.15
    xbutton  = 0.3
    dybutton = hbutton+0.01
    panelbot = 0.0
    controlh = panelbot + len(params)*dyslider
    if plotbutton: controlh += dybutton
    controltop = panelbot + controlh
    bmargin  = 0.15
    
    # generate figure
    if fig is None: fig = plt.figure()
    fig.subplots_adjust(top=0.95,bottom=controltop+bmargin)

    # Set the initial values
    indexinit = np.zeros(len(params),dtype=int)
    if parstart is not None:
        for i in range(len(params)):
            if parstart[i] in params[i]:
                idx = np.where(np.array(params[i])==parstart[i])[0]
                if len(idx)>0:
                    indexinit[i] = idx[0]
            else:
                if params[i][-1]>params[i][0]:
                    idx = np.where(np.array(params[i])<parstart[i])[0]
                    if len(idx)>0:
                        indexinit[i] = idx[-1]
                else:
                    idx = np.where(np.array(params[i])>parstart[i])[0]
                    if len(idx)>0:
                        indexinit[i] = idx[0]
    if iparstart is not None:
        indexinit[:] = iparstart[:]

    # select first image
    par = []
    for i in range(len(params)):
        par.append(params[i][indexinit[i]])
    if fixedpar is not None:
        f = func(x,par,fixedpar=fixedpar)
    else:
        f = func(x,par)

    # set range
    if ymin is None: ymin = f.min()
    if ymax is None: ymax = f.max()
    
    # display function(s)
    if ax is None:      ax       = plt.axes(xlim=(x.min(),x.max()),ylim=(ymin,ymax))
    if axmodel is None:
        if len(f.shape)==1:
            # Normal case: a single model function
            axmodel, = ax.plot(x,f,**kwargs)
        else:
            # Special case: multiple model functions: f[imodel,:]
            assert len(f.shape)==2, 'Model returns array with more than 2 dimensions. No idea what to do.'
            axmodel = []
            for i in range(f.shape[0]):
                axm, = ax.plot(x,f[i,:],**kwargs)
                axmodel.append(axm)
            
    sliders = []
    for i in range(len(params)):
    
        # define slider
        axcolor = 'lightgoldenrodyellow'
        axs = fig.add_axes([xslider, controltop-i*dyslider, xslider+wslider, hslider], facecolor=axcolor)

        if parnames is not None:
            name = parnames[i]
        else:
            name = 'Parameter {0:d}'.format(i)

        slider = Slider(axs, name, 0, len(params[i]) - 1,
                    valinit=indexinit[i], valfmt='%i')
        sliders.append(slider)

    if plotbutton:
        axb = fig.add_axes([xbutton, panelbot+0.2*hbutton, xbutton+wbutton, hbutton])
        pbutton = Button(axb,'Plot')
    else:
        pbutton = None

    class callbackplot(object):
        def __init__(self,x,func,params,sliders,pbutton=None,fixedpar=None,ipar=None):
            self.x        = x
            self.func     = func
            self.params   = params
            self.sliders  = sliders
            self.pbutton  = pbutton
            self.fixedpar = fixedpar
            self.parunits = parunits
            self.paramsalt= paramsalt
            self.altformat= altformat
            self.closed   = False
            if ipar is None:
                self.ipar = np.zeros(len(sliders),dtype=int)
            else:
                self.ipar = ipar
        def handle_close(self,event):
            self.closed   = True
        def myreadsliders(self):
            for isl in range(len(self.sliders)):
                ind = int(self.sliders[isl].val)
                self.ipar[isl]=ind
            par = []
            for i in range(len(self.ipar)):
                ip = self.ipar[i]
                value = self.params[i][ip]
                par.append(value)
                name = self.sliders[i].label.get_text()
                if '=' in name:
                    namebase = name.split('=')[0]
                    if self.paramsalt is not None:
                        vls  = "{0:" + self.altformat + "}"
                        name = namebase + "= " + vls.format(self.paramsalt[i][ip])
                    else:
                        if self.parunits is not None:
                            valunit = self.parunits[i]
                        else:
                            valunit = 1.0
                        name = namebase + "= {0:13.6e}".format(value/valunit)
                    self.sliders[i].label.set_text(name)
            return par
        def myreplot(self,par):
            x = self.x
            if self.fixedpar is not None:
                f = self.func(x,par,fixedpar=self.fixedpar)
            else:
                f = self.func(x,par)
            if len(f.shape)==1:
                axmodel.set_data(x,f)
            else:
                for i in range(f.shape[0]):
                    axmodel[i].set_data(x,f[i,:])
            plt.draw()
        def mysupdate(self,event):
            par = self.myreadsliders()
            if self.pbutton is None: self.myreplot(par)
        def mybupdate(self,event):
            par = self.myreadsliders()
            if self.pbutton is not None: self.pbutton.label.set_text('Computing...')
            plt.pause(0.01)
            self.myreplot(par)
            if self.pbutton is not None: self.pbutton.label.set_text('Plot')

    mcb = callbackplot(x,func,params,sliders,pbutton=pbutton,fixedpar=fixedpar,ipar=indexinit)

    mcb.mybupdate(0)

    if plotbutton:
        pbutton.on_clicked(mcb.mybupdate)
    for s in sliders:
        s.on_changed(mcb.mysupdate)

    fig._mycallback    = mcb

    if block:
        plt.show(block=True)
    if returnipar:
        return mcb.ipar
        

def interactive_curve(t, func, params, xmin=None, xmax=None, ymin=None, ymax=None, parnames=None, parunits=None, fig=None, ax=None, axmodel=None, parstart=None, iparstart=None, plotbutton=False, fixedpar=None, returnipar=False, block=False, **kwargs):
    """
    Plot the 2-D curve x,y = func(t) with parameters given by the params
    list of lists. 

    ARGUMENTS:
      t          Array of t values
      func       Function func(x,params)
      params     List of parameters, but with each parameter value
                 here given as a list of possible values.

    OPTIONAL ARGUMENTS:
      xmin       Set horizontal axis lower limit
      xmax       Set horizontal axis upper limit
      ymin       Set vertical axis lower limit
      ymax       Set vertical axis upper limit
      parnames   Names of the params, e.g. ['A', 'omega']
                 If the parnames have an '=' sign (e.g. ['A = ', 'omega = '])
                 then the value of the parameters are written out.
      parunits   If set, a list of values by which the parameter values are divided
                 before being printed on the widget (only if parnames have '=').
                 It only affects the printing next to the sliders, and has no 
                 other effect.
      fig        A pre-existing figure
      ax         A pre-existing axis
      parstart   If set, set the sliders initially close to these values
      iparstart  If set, set the slider index values initially to these values
                 (note: iparstart is an alternative to parstart)
      returnipar If True, then return ipar
      block      If True, then wait until window is closed

    EXAMPLE 1 (one ellipse):
    from interactive_plot import *
    def func(t,param): 
        x = param[0]*np.cos(t)
        y = param[1]*np.sin(t)
        csw = np.cos(param[2])
        snw = np.sin(param[2])
        return csw*x-snw*y,snw*x+csw*y
    t      = np.linspace(0,2*np.pi,100)
    params = [np.linspace(0.1,1.,30),np.linspace(0.1,1.,30),np.linspace(0.,np.pi,30)]
    interactive_curve(t, func, params, xmax=1., xmin=-1., ymax=1., ymin=-1., parnames=['Ax = ','Ay = ','omega = '],iparstart=[10,15,12])

    EXAMPLE 1-a (With plotting button instead of automatic replot; useful for heavier models):
    from interactive_plot import *
    def func(t,param): 
        x = param[0]*np.cos(t)
        y = param[1]*np.sin(t)
        csw = np.cos(param[2])
        snw = np.sin(param[2])
        return csw*x-snw*y,snw*x+csw*y
    t      = np.linspace(0,2*np.pi,100)
    params = [np.linspace(0.1,1.,30),np.linspace(0.1,1.,30),np.linspace(0.,np.pi,30)]
    interactive_curve(t, func, params, xmax=1., xmin=-1., ymax=1., ymin=-1., parnames=['Ax = ','Ay = ','omega = '],iparstart=[10,15,12],plotbutton=True)

    EXAMPLE 2 (two ellipses):
    import numpy as np
    import matplotlib.pyplot as plt
    from interactive_plot import *
    def func(t,param): 
        x = param[0]*np.cos(t)
        y = param[1]*np.sin(t)
        csw = np.cos(param[2])
        snw = np.sin(param[2])
        return np.vstack((csw*x-snw*y,-csw*x-snw*y)),np.vstack((snw*x+csw*y,snw*x+csw*y))
    t      = np.linspace(0,2*np.pi,100)
    params = [np.linspace(0.1,1.,30),np.linspace(0.1,1.,30),np.linspace(0.,np.pi,30)]
    fig    = plt.figure(1)
    ax     = plt.axes(xlim=(-1.2,1.2),ylim=(-1.2,1.2))
    x,y    = func(t,[1.,1.,1.])
    axm0,  = ax.plot(x[0,:],y[0,:],'--',label='left')
    axm1,  = ax.plot(x[1,:],y[1,:],':',label='right')
    axmodel= [axm0,axm1]
    interactive_curve(t, func, params, xmax=1., xmin=-1., ymax=1., ymin=-1., parnames=['Ax = ','Ay = ','omega = '],iparstart=[10,15,12], fig=fig, ax=ax, axmodel=axmodel)

    EXAMPLE 3 (as example 2, but now each ellipse in its own panel):
    import numpy as np
    import matplotlib.pyplot as plt
    from interactive_plot import *
    def func(t,param): 
        x = param[0]*np.cos(t)
        y = param[1]*np.sin(t)
        csw = np.cos(param[2])
        snw = np.sin(param[2])
        return np.vstack((csw*x-snw*y,-csw*x-snw*y)),np.vstack((snw*x+csw*y,snw*x+csw*y))
    t      = np.linspace(0,2*np.pi,100)
    params = [np.linspace(0.1,1.,30),np.linspace(0.1,1.,30),np.linspace(0.,np.pi,30)]
    fig, axes = plt.subplots(nrows=2)
    axes[0].set_xlim((-1.2,1.2))
    axes[0].set_ylim((-1.2,1.2))
    axes[1].set_xlim((-1.2,1.2))
    axes[1].set_ylim((-0.8,0.8))
    x,y    = func(t,[1.,1.,1.])
    axm0,  = axes[0].plot(x[0,:],y[0,:],'--',label='left')
    axm1,  = axes[1].plot(x[1,:],y[1,:],':',label='right')
    axmodel= [axm0,axm1]
    interactive_curve(t, func, params, xmax=1., xmin=-1., ymax=1., ymin=-1., parnames=['Ax = ','Ay = ','omega = '],iparstart=[10,15,12], fig=fig, ax=axes[0], axmodel=axmodel)
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons

    # Compute spacing of plot, sliders and button
    hslider  = 0.03
    nslidrscl= 6
    if(len(params)>nslidrscl):
        hslider *= float(nslidrscl)/len(params)
    dyslider = hslider*(4./3.)
    xslider  = 0.3
    wslider  = 0.3
    hbutton  = 0.06
    wbutton  = 0.15
    xbutton  = 0.3
    dybutton = hbutton+0.01
    panelbot = 0.0
    controlh = panelbot + len(params)*dyslider
    if plotbutton: controlh += dybutton
    controltop = panelbot + controlh
    bmargin  = 0.15
    
    # generate figure
    if fig is None: fig = plt.figure()
    fig.subplots_adjust(top=0.95,bottom=controltop+bmargin)

    # Set the initial values
    indexinit = np.zeros(len(params),dtype=int)
    if parstart is not None:
        for i in range(len(params)):
            if parstart[i] in params[i]:
                idx = np.where(np.array(params[i])==parstart[i])[0]
                if len(idx)>0:
                    indexinit[i] = idx[0]
            else:
                if params[i][-1]>params[i][0]:
                    idx = np.where(np.array(params[i])<parstart[i])[0]
                    if len(idx)>0:
                        indexinit[i] = idx[-1]
                else:
                    idx = np.where(np.array(params[i])>parstart[i])[0]
                    if len(idx)>0:
                        indexinit[i] = idx[0]
    if iparstart is not None:
        indexinit[:] = iparstart[:]

    # select first image
    par = []
    for i in range(len(params)):
        par.append(params[i][indexinit[i]])
    if fixedpar is not None:
        x, y = func(t,par,fixedpar=fixedpar)
    else:
        x, y = func(t,par)

    # set range
    if xmin is None: xmin = x.min()
    if xmax is None: xmax = x.max()
    if ymin is None: ymin = y.min()
    if ymax is None: ymax = y.max()
    
    # display function
    if ax is None: ax   = plt.axes(xlim=(xmin,xmax),ylim=(ymin,ymax))
    if axmodel is None:
        if len(x.shape)==1:
            # Normal case: a single model function
            assert len(x.shape)==1, 'Cannot have multiple y and single x'
            axmodel, = ax.plot(x,y,**kwargs)
        else:
            # Special case: multiple model functions: f[imodel,:]
            assert len(x.shape)==2, 'Model returns array with more than 2 dimensions. No idea what to do.'
            assert len(y.shape)==2, 'Cannot have multiple x and single y'
            axmodel = []
            for i in range(x.shape[0]):
                axm, = ax.plot(x[i,:],y[i,:],**kwargs)
                axmodel.append(axm)
    
    sliders = []
    for i in range(len(params)):
    
        # define slider
        axcolor = 'lightgoldenrodyellow'
        axs = fig.add_axes([xslider, controltop-i*dyslider, xslider+wslider, hslider], facecolor=axcolor)

        if parnames is not None:
            name = parnames[i]
        else:
            name = 'Parameter {0:d}'.format(i)
            
        slider = Slider(axs, name, 0, len(params[i]) - 1,
                    valinit=indexinit[i], valfmt='%i')
        sliders.append(slider)

    if plotbutton:
        axb = fig.add_axes([xbutton, panelbot+0.2*hbutton, xbutton+wbutton, hbutton])
        pbutton = Button(axb,'Plot')
    else:
        pbutton = None

    class callbackcurve(object):
        def __init__(self,t,func,params,sliders,pbutton=None,fixedpar=None,ipar=None):
            self.t        = t
            self.func     = func
            self.params   = params
            self.sliders  = sliders
            self.pbutton  = pbutton
            self.fixedpar = fixedpar
            self.parunits = parunits
            self.closed   = False
            if ipar is None:
                self.ipar = np.zeros(len(sliders),dtype=int)
            else:
                self.ipar = ipar
        def handle_close(self,event):
            self.closed   = True
        def myreadsliders(self):
            for isl in range(len(self.sliders)):
                ind = int(self.sliders[isl].val)
                self.ipar[isl]=ind
            par = []
            for i in range(len(self.ipar)):
                ip = self.ipar[i]
                value = self.params[i][ip]
                par.append(value)
                name = self.sliders[i].label.get_text()
                if '=' in name:
                    namebase = name.split('=')[0]
                    if self.parunits is not None:
                        valunit = self.parunits[i]
                    else:
                        valunit = 1.0
                    name = namebase + "= {0:13.6e}".format(value/valunit)
                    self.sliders[i].label.set_text(name)
            return par
        def myreplot(self,par):
            t = self.t
            if self.fixedpar is not None:
                x,y = self.func(t,par,fixedpar=self.fixedpar)
            else:
                x,y = self.func(t,par)
            if len(x.shape)==1:
                axmodel.set_data(x,y)
            else:
                for i in range(x.shape[0]):
                    axmodel[i].set_data(x[i,:],y[i,:])
            plt.draw()
        def mysupdate(self,event):
            par = self.myreadsliders()
            if self.pbutton is None: self.myreplot(par)
        def mybupdate(self,event):
            par = self.myreadsliders()
            if self.pbutton is not None: self.pbutton.label.set_text('Computing...')
            plt.pause(0.01)
            self.myreplot(par)
            if self.pbutton is not None: self.pbutton.label.set_text('Plot')

    mcb = callbackcurve(t,func,params,sliders,pbutton=pbutton,fixedpar=fixedpar,ipar=indexinit)
            
    mcb.mybupdate(0)
        
    if plotbutton:
        pbutton.on_clicked(mcb.mybupdate)
    for s in sliders:
        s.on_changed(mcb.mysupdate)

    fig._mycallback    = mcb
    
    if block:
        plt.show(block=True)
    if returnipar:
        return mcb.ipar
