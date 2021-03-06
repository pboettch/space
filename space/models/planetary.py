import numpy as np
import pandas as pd

import sys
sys.path.append('.')
from .. import utils
from ..coordinates import coordinates as coords
from ..smath import resolve_poly2



def _checking_angles(theta, phi):

    if isinstance(theta, np.ndarray) and isinstance(theta, np.ndarray) and len(theta.shape) > 1 and len(phi.shape) > 1:
        return np.meshgrid(theta, phi)
    return theta, phi


def _formisano1979(theta, phi, **kwargs):
    a11,a22,a33,a12,a13,a23,a14,a24,a34,a44 = kwargs["coefs"]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    A = a11 * ct**2 + st**2 * (a22* cp**2 + a33 * sp**2) \
        + ct * st * (a12 * cp + a13 * sp) + a23 * st**2 * cp * sp
    B = a14*ct + st*(a24*cp + a34*sp)
    C = a44
    D = B**2 - 4*A*C
    return (-B + np.sqrt(D))/(2*A)


def formisano1979(theta, phi, **kwargs):
    '''
    Formisano 1979 magnetopause model. Give the default position of the magnetopause.
    function's arguments :
        - theta : angle in radiant, can be int, float or array (1D or 2D)
        - phi   : angle in radiant, can be int, float or array (1D or 2D)
    kwargs:
        - boundary  : "magnetopause", "bow_shock"
        - base  : can be "cartesian" (default) or "spherical"

    information : to get a particular point theta and phi must be an int or a float
    (ex : the nose of the boundary is given with the input theta=0 and phi=0). If a plan (2D) of
    the boundary is wanted one of the two angle must be an array and the other one must be
    an int or a float (ex : (XY) plan with Z=0 is given with the input
    theta=np.linespace(-np.pi/2,np.pi/2,100) and phi=0). To get the 3D boundary theta and phi
    must be given an array, if it is two 1D array a meshgrid will be performed to obtain a two 2D array.

    return : X,Y,Z (base="cartesian")or R,theta,phi (base="spherical") depending on the chosen base
    '''

    if kwargs["boundary"] == "magnetopause":
        coefs = [0.65,1,1.16,0.03,-0.28,-0.11,21.41,0.46,-0.36,-221]
    elif kwargs["boundary"] == "bow_shock":
        coefs = [0.52, 1, 1.05, 0.13, -0.16, -0.08, 47.53, -0.42, 0.67, -613]
    else:
        raise ValueError("boundary: {} not allowed".format(kwargs["boundary"]))

    theta, phi = _checking_angles(theta, phi)
    r          =  _formisano1979(theta, phi, coefs = coefs)
    base       = kwargs.get("base", "cartesian")
    if base == "cartesian":
        return coords.spherical_to_cartesian(r, theta, phi)
    elif base == "spherical":
        return r, theta, phi
    raise ValueError("unknown base '{}'".format(kwargs["base"]))



def mp_formisano1979(theta, phi, **kwargs):
    return formisano1979(theta, phi, boundary="magnetopause", **kwargs)

def bs_formisano1979(theta, phi, **kwargs):
    return formisano1979(theta, phi, boundary="bow_shock", **kwargs)




def Fairfield1971(x, args):

    '''
    Fairfield 1971 : Magnetopause and Bow shock models. Give positions of the boudaries in plans (XY) with Z=0 and (XZ) with Y=0.
    function's arguments :
        - x :  X axis (array) in Re (earth radii)
        - args : coefficients Aij are determined from many boundary crossings and they depend on upstream conditions.

        --> Default parameter for the bow shock and the magnetopause respectively are :
            default_bs_fairfield = [0.0296,-0.0381,-1.28,45.644,-652.1]
            default_mp_fairfield = [-0.0942,0.3818,0.498,17.992,-248.12]

     return : DataFrame (Pandas) with the position (X,Y,Z) in Re of the wanted boudary to plot (XY) and (XZ) plans.
    '''


    A,B,C,D,E = args[0],args[1],args[2],args[3],args[4]

    a = 1
    b = A*x + C
    c = B*x**2 + D*x + E

    delta = b**2-4*a*c

    ym = (-b - np.sqrt(delta))/(2*a)
    yp = (-b + np.sqrt(delta))/(2*a)

    pos=pd.DataFrame({'X' : np.concatenate([x, x[::-1]]),
                      'Y' : np.concatenate([yp, ym[::-1]]),
                      'Z' : np.concatenate([yp, ym[::-1]]),})

    return pos.dropna()






def bs_Jerab2005( Np, V, Ma, B, gamma=2.15 ):

    '''
    Jerab 2005 Bow shock model. Give positions of the box shock in plans (XY) with Z=0 and (XZ) with Y=0 as a function of the upstream solar wind.
    function's arguments :
        - Np : Proton density of the upstream conditions
        - V  : Speed of the solar wind
        - Ma : Alfven Mach number
        - B  : Intensity of interplanetary magnetic field
        - gamma : Polytropic index ( default gamma=2.15)


        --> mean parameters :  Np=7.35, V=425.5, Ma=11.23, B=5.49

     return : DataFrame (Pandas) with the position (X,Y,Z) in Re of the bow shock to plot (XY) and (XZ) plans.
    '''

    def make_Rav(theta,phi):
        a11 = 0.45
        a22 = 1
        a33 = 0.8
        a12 = 0.18
        a14 = 46.6
        a24 = -2.2
        a34 = -0.6
        a44 = -618

        a = a11*np.cos(theta)**2 + np.sin(theta)**2 *( a22*np.cos(phi)**2 + a33*np.sin(phi)**2 )
        b = a14*np.cos(theta) +  np.sin(theta) *( a24*np.cos(phi) + a34*np.sin(phi) )
        c = a44

        delta = b**2 -4*a*c

        R = (-b + np.sqrt(delta))/(2*a)
        return R



    C = 91.55
    D = 0.937*(0.846 + 0.042*B )
    R0 = make_Rav(0,0)

    theta = np.linspace(0,2.5,200)
    phi = [np.pi,0]

    x,y= [],[]

    for p in phi:
        Rav = make_Rav(theta,p)
        K = ((gamma-1)*Ma**2+2)/((gamma+1)*(Ma**2-1))
        R = (Rav/R0)*(C/(Np*V**2)**(1/6))*(1+ D*K)

        x = np.concatenate([x, R*np.cos(theta)])
        y = np.concatenate([y, R*np.sin(theta)*np.cos(p)])


    pos = pd.DataFrame({'X' : x , 'Y' : y, 'Z' : y})

    return pos.sort_values('Y')


def mp_shue1998(theta, phi, **kwargs):
    '''
    Shue 1998 Magnetopause model.

     Returns the MP distance for given theta,
     dynamic pressure (in nPa) and Bz (in nT).

    - theta : Angle from the x axis (model is cylindrical symmetry)
         example : theta = np.arange( -np.pi+0.01, np.pi-0.01, 0.001)
    - phi   : unused, azimuthal angle, to get the same interface as all models

    kwargs:
    - "pdyn": Dynamic Pressure in nPa
    - "Bz"  : z component of IMF in nT

    '''

    Pd = kwargs.get("pdyn", 2)
    Bz = kwargs.get("Bz", 1)

    r0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Pd**(-1./6.6)
    a = (0.58-0.007*Bz)*(1+0.024*np.log(Pd))
    r = r0*(2./(1+np.cos(theta)))**a


    base = kwargs.get("base", "cartesian")
    if base == "cartesian":
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = r * np.sin(theta)
        return x, y, z
    elif base == "cylindrical":
        return r, theta
    raise ValueError("unknown base '{}'".format(kwargs["base"]))



def MP_Lin2010(phi_in ,th_in, Pd, Pm, Bz, tilt=0.):
    ''' The Lin 2010 Magnetopause model. Returns the MP distance for a given
    azimuth (phi), zenith (th), solar wind dynamic and magnetic pressures (nPa)
    and Bz (in nT).
    * th: Zenith angle from positive x axis (zenith) (between 0 and pi)
    * phi: Azimuth angle from y axis, about x axis (between 0 abd 2 pi)
    * Pd: Solar wind dynamic pressure in nPa
    * Pm: Solar wind magnetic pressure in nPa
    * tilt: Dipole tilt
    '''
    a = [12.544,
         -0.194,
         0.305,
         0.0573,
         2.178,
         0.0571,
         -0.999,
         16.473,
         0.00152,
         0.382,
         0.0431,
         -0.00763,
         -0.210,
         0.0405,
         -4.430,
         -0.636,
         -2.600,
         0.832,
         -5.328,
         1.103,
         -0.907,
         1.450]

    arr = type(np.array([]))

    if(type(th_in) == arr):
        th = th_in.copy()
    else:
        th = th_in

    if(type(phi_in) == arr):
        phi = phi_in.copy()
    else:
        phi = phi_in

    el = th_in < 0.
    if(type(el) == arr):
        if(el.any()):
            th[el] = -th[el]

            if(type(phi) == type(arr)):
                phi[el] = phi[el]+np.pi
            else:
                phi = phi*np.ones(th.shape)+np.pi*el
    else:
        if(el):
            th = -th
            phi = phi+np.pi

    P = Pd+Pm

    def exp2(i):
        return a[i]*(np.exp(a[i+1]*Bz)-1)/(np.exp(a[i+2]*Bz)+1)

    def quad(i, s):
        return a[i]+s[0]*a[i+1]*tilt+s[1]*a[i+2]*tilt**2

    r0 = a[0]*P**a[1]*(1+exp2(2))

    beta = [a[6] + exp2(7),
            a[10],
            quad(11, [1, 0]),
            a[13]]

    f = np.cos(0.5*th)+a[5]*np.sin(2*th)*(1-np.exp(-th))
    s = beta[0]+beta[1]*np.sin(phi)+beta[2]*np.cos(phi)+beta[3]*np.cos(phi)**2
    f = f**(s)

    c = {}
    d = {}
    TH = {}
    PHI = {}
    e = {}
    for i, s in zip(['n', 's'], [1, -1]):
        c[i] = a[14]*P**a[15]
        d[i] = quad(16, [s, 1])
        TH[i] = quad(19, [s, 0])
        PHI[i] = np.cos(th)*np.cos(TH[i])
        PHI[i] = PHI[i] + np.sin(th)*np.sin(TH[i])*np.cos(phi-(1-s)*0.5*np.pi)
        PHI[i] = np.arccos(PHI[i])
        e[i] = a[21]
    r = f*r0

    Q = c['n']*np.exp(d['n']*PHI['n']**e['n'])
    Q = Q + c['s']*np.exp(d['s']*PHI['s']**e['s'])

    return r+Q


_models = {"mp_shue": mp_shue1998,
           "mp_formisano1979": mp_formisano1979,
           "bs_formisano1979": bs_formisano1979,
           "bs_jerab": bs_Jerab2005}



def _interest_points(model, **kwargs):
    dup = kwargs.copy()
    dup["base"] = "cartesian"
    x = model(0, 0, **dup)[0]
    y = model(np.pi / 2, 0, **dup)[1]
    xf = x - y ** 2 / (4 * x)
    return x, y, xf


def _parabolic_approx(theta, phi, x, xf, **kwargs):
    theta, phi = _checking_angles(theta, phi)
    K = x - xf
    a = np.sin(theta) ** 2
    b = 4 * K * np.cos(theta)
    c = -4 * K * x
    r = resolve_poly2(a, b, c)[0]
    return coords.BaseChoice(kwargs.get("base", "cartesian"), r, theta, phi)



def check_parabconfoc(func):
    def wrapper(self, theta, phi, **kwargs):
        kwargs["parabolic"] = kwargs.get("parabolic", False)
        kwargs["confocal"] = kwargs.get("confocal", False)
        if kwargs["parabolic"] is False and kwargs["confocal"] is True:
            raise ValueError("cannot be confocal if not parabolic")
        return func(self, theta, phi, **kwargs)
    return wrapper


class Magnetosheath:
    def __init__(self, **kwargs):
        self._magnetopause = _models[kwargs.get("magnetopause", "shue")]
        self._bow_shock = _models[kwargs.get("bow_shock", "jerab")]


    @check_parabconfoc
    def magnetopause(self, theta, phi, **kwargs):
        if kwargs["parabolic"]:
            return self._parabolize(theta, phi, **kwargs)[0]
        else:
            return self._magnetopause(theta, phi, **kwargs)


    @check_parabconfoc
    def bow_shock(self, theta, phi, **kwargs):
        if kwargs["parabolic"]:
            return self._parabolize(theta, phi, **kwargs)[1]
        else:
            return self._bow_shock(theta, phi, **kwargs)

    @check_parabconfoc
    def boundaries(self, theta, phi, **kwargs):
        if kwargs["parabolic"]:
            return self._parabolize(theta, phi, **kwargs)
        else:
            return self._magnetopause(theta, phi, **kwargs),\
                   self._bow_shock(theta, phi, **kwargs)


    def _parabolize(self, theta, phi, **kwargs):
        xmp, y, xfmp = _interest_points(self._magnetopause, **kwargs)
        xbs, y, xfbs = _interest_points(self._bow_shock, **kwargs)
        if kwargs.get("confocal", False) is True:
            xfmp = xmp / 2
            xfbs = xmp / 2
        mp_coords = _parabolic_approx(theta, phi, xmp, xfmp, **kwargs)
        bs_coords = _parabolic_approx(theta, phi, xbs, xfbs, **kwargs)
        return mp_coords, bs_coords
