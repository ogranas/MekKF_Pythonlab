import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import pi

def dtheta(state, t, I, p, E, gamma):
    """
    A function used to evolve the state of a torsion
    oscillator composed of a dipolar molecule in an
    electric field, with damping, by solving:
    dtheta/dt = omega
    domega/dt=-(gamma/I)*omega - (p*E/I)*sin(theta)
    
    Parameters:
    ----------
    state: [float, float]
        The state of the evolving system, [theta, omega]
    t: [float]
        Array with discretized time-points
    I: [float]
        Moment of inertia of the molecule 
    p: [float]
        Dipole moment
    E: [float]
        Electric field strength
    gamma: [float]
        damping
        
    returns state
    """
    theta, omega = state[0], state[1]
    dtheta =omega
    domega = -(gamma / I )* omega - (p*E/I)*np.sin(theta)
    return [dtheta, domega]

def dy(y, t, gamma, w0, drive_amp, drive_w):
    """
    A function used to evolve the state of a linear
    harmonic oscillator by solving returning
    dx/dt = p
    dp/dt = -2 gamma p - w0^2 x + w0^2 A cos(wd t)

    Parameters
    ----------
    y: [float, float]
        The state of the evolving system, [x, p]
    t: [float]
        Array with discretized time-points
    gamma: [float]
        Damping factor
    w0: [float]
        Resonace frequency
    drive_amp: [float]
        Amplitude of the drive forcing the oscillator
    drive_w: [float]
        Angular frequency of the drive forcing the oscillator
    """

    x, p = y[0], y[1]
    dx = p
    dp = -2 * gamma * p - w0**2 * x + w0**2 * drive_amp*np.cos(drive_w*t)
    return [dx, dp]

def solve_torsion_oscillator(t, theta_0, omega_0, I, p, E, gamma=0.0):
    """
    Simplifying the interface to odeint when solving the torsion oscillator
        
    Parameters
    ----------   
    t: [float]
        Array with discretized time-points
    theta_0: [float]
        Initial condition for the angle
    theta_0: [float]
        Initial condition for the angular frequency    
    I: [float]
        Moment of inertia of the molecule 
    p: [float]
        Dipole moment
    E: [float]
        Electric field strength
    gamma: [float], optional
        damping
        
    returns trajectory with theta and omega as function of time
    
    """
    y0 = [theta_0, omega_0]
    y1 = odeint(dtheta, y0, t, args = (I, p, E, gamma))
    return y1[:,0], y1[:,1]

def solve_linear_harmonic_oscillator(t, initial_ampl, initial_velocity, resonance_freq, damping = 0.0, drive_ampl = 0.0, drive_ang_freq = 0.0):
    """
    Simplifying the interface to odeint when solving the driven linear oscillator
        
    Parameters
    ----------   
    t: [float]
        Array with discretized time-points
    initial_ampl: [float]
        Initial condition for amplitude
    initial_velocity: [float]
        Initial condition for the velocity 
    resonance_freq: [float]
        Resonance frequency of the oscillator
    damping: [float], optional
        Damping factor
    drive_amp: [float], optional
        Amplitude of the drive forcing the oscillator
    drive_w: [float], optional
        Angular frequency of the drive forcing the oscillator
        
    retorns solution as trajectories of position and velocity
    """


    x = initial_ampl
    p = initial_velocity
    y0 = [initial_ampl, initial_velocity]
    y1 = odeint(dy, y0, t, args=(damping, resonance_freq, drive_ampl, drive_ang_freq)) # under damped
    return y1[:,0], y1[:,1]

def measure_molecule(time, E):
    """
    Secret functionality
    """
    
    theta_0 = (np.pi*0.5)*(np.random.rand()-0.5)
    omega_0 = 0.0
    Da_to_kg=1.661e-17
    D_to_Cm=3.336e-30

    m_ubiquitin = 8566*Da_to_kg
    m_lysozyme = 14305*Da_to_kg
    m_myoglobin = 16950*Da_to_kg

    p_ubiquitin = 189*D_to_Cm
    p_lysozyme = 218*D_to_Cm
    p_myoglobin = 225*D_to_Cm
    
    molecule = 1 # We only measure one... 
    rho=1 #kg/l
    
    def mass_to_radius(mass): #density is one, protein is globular and homogeneous
        r=np.power(
            np.divide(3.0*mass,rho*4*np.pi),1.0/3.0)
        return r
        
    def mass_to_I(mass): #density is one, protein is globular and homogeneous
        r = mass_to_radius(mass)
        return 0.4*mass*r*r
    
    if (molecule == 1):
        m=m_lysozyme
        I = mass_to_I(m)
        p = p_lysozyme
        r = mass_to_radius(m)
        gamma = I*1e-3
    else:
        print('No data for molecule number {0}'.format(molecule))

    return solve_torsion_oscillator(time, theta_0, omega_0, I, p, E, gamma)[0]


def measure_IR(bond, time, Ad, Wd):
    """
    Secret functionality
    """
    x0 = (np.pi*0.5)*(np.random.rand()-0.5)
    v0 = 0.0
    if (bond == 1):
        w0 = 1.2
        gamma = 0.1
    elif (bond == 2):
        w0 = 0.8
        gamma = 0.15
    elif (bond == 3):
        w0 = 1.2
        gamma = 0.5
    elif (bond == 4):
        w0 = 2.4
        gamma = 0.2
    else:
        print('No data for molecule number {0}'.format(molecule))

    return solve_linear_harmonic_oscillator(time, x0, v0, w0, gamma, Ad, Wd)[0]

def fit_steady_state(time,data,p=None, wd=None):
    from scipy.optimize import curve_fit
    """
    Fit sin to the input data
    return fitted function and
    fitting parameters:
    time: [float]
        Array with discretized time-points
    data: [float]
        Array with data for fit
    p: [float]
        Phase, optional, if not set it is fitted.
    wd: [float]
        Frequency, if not set it is fitted. 
        
    Returns:
    fitted_function: [float]
        array with fitted function on the supplied time-array
    A: [float]
        Amplitude
    w: [float]
        angular frequency
    p: [float]
        phase
    c: [float]
        offset
    """
    
    def guess_fq(tt,yy):
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        return abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    
    def guess_amp(tt,yy):
        return np.std(yy) * 2.**0.5

    if ((wd is None) and (p is None)):
        def sin_func(t, A, w, p):
            return A*np.sin(w*t+p)   
 
        tt = np.array(time)
        yy = np.array(data)

        guess = np.array([guess_amp(tt,yy), 2.*np.pi*guess_fq(tt,yy), 0.])
        
        try:
            popt, popc = curve_fit(sin_func, tt, yy, p0=guess, bounds=(0.0,[np.inf, np.inf,1.99*np.pi]))
        except:
            print("Curve fit failed to converge, returning initial guess")
            popt = guess
        A, w, p = popt
            
    elif ((wd is None) and (p is not None)):
        def sin_func(t, A, w, p):
            return A*np.sin(w*t+p)   
 
        tt = np.array(time)
        yy = np.array(data)

        guess = np.array([guess_amp(tt,yy), 2.*np.pi*guess_fq(tt,yy), 0.])
        
        try:
            popt, popc = curve_fit(sin_func, tt, yy, p0=guess, bounds=([0.0, 0.0, -np.pi],[np.inf, np.inf, np.pi]))
        except:
            print("Curve fit failed to converge, returning initial guess")
            popt = guess
        A, w = popt
            
    elif ((wd is not None) and (p is None)):
        def sin_func(t, A, p):
            return A*np.sin(wd*t+p)   
 
        tt = np.array(time)
        yy = np.array(data)

        guess = np.array([guess_amp(tt,yy), 0.])
        
        try:
            popt, popc = curve_fit(sin_func, tt, yy, p0=guess, bounds=([0.0, -np.pi], [np.inf, np.pi]))
        except:
            print("Curve fit failed to converge, returning initial guess")
            popt = guess
        A,  p = popt
        w = wd

    else: # both set
        def sin_func(t, A):
            return A*np.sin(wd*t+p)   
 
        tt = np.array(time)
        yy = np.array(data)

        guess = np.array([guess_amp(tt,yy)])
        
        try:
            popt, popc = curve_fit(sin_func, tt, yy, p0=guess, bounds=(0.0,np.inf))
        except:
            print("Curve fit failed to converge, returning initial guess")
            popt = guess
        A = popt  
        w=wd
        p=p
                         

    # return sin_func(tt, A, w, p), [A, w, p]
    return A*np.sin(w*tt+p), [A, w, p]

