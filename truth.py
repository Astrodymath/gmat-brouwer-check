"""Reference model for v04-00 tests: Cartesian zonal problem J2, J3, J4, DOP853 (rtol 1e-13, atol 1e-12 km, km/s).
Units km, s. Constants as in the paper (mu, Re, J2) plus J3, J4 (EGM2008, unnormalised)."""
import numpy as np
from scipy.integrate import solve_ivp
MU=398600.4418; RE=6378.137
J2=1.08262668e-3; J3=-2.53265649e-6; J4=-1.61962159e-6

def accel(x, j2=J2, j3=J3, j4=J4):
    """acceleration of -grad(U), U = -mu/r [1 - sum J_n (R/r)^n P_n(z/r)]  (x: (...,3))"""
    x=np.asarray(x,float); r2=np.sum(x*x,-1); r=np.sqrt(r2); u=x[...,2]/r
    # potential V = mu/r * sum_n J_n (R/r)^n P_n(u)  is the disturbing potential energy P (paper sign)
    # a = -grad(-mu/r + P) ; compute grad P analytically
    a=-MU*x/r[...,None]**3
    def addn(Jn,n,Pn,dPn):
        nonlocal a
        if Jn==0: return
        k=MU*Jn*RE**n/r**(n+1)          # P_n term = k * Pn(u)
        # grad (k Pn(u)) = Pn * dk/dr * x/r + k dPn(u) * grad u ; grad u = (e_z - u x/r)/r
        dk=-(n+1)*k/r
        g=(Pn*dk)[...,None]*x/r[...,None]
        gu=-(u/r)[...,None]*x/r[...,None]; gu[...,2]+=1/r
        g+= (k*dPn)[...,None]*gu
        a=a-g
    addn(j2,2,0.5*(3*u**2-1),3*u)
    addn(j3,3,0.5*(5*u**3-3*u),0.5*(15*u**2-3))
    addn(j4,4,(35*u**4-30*u**2+3)/8,(140*u**3-60*u)/8)
    return a

def potential_P(x, j2=J2, j3=J3, j4=J4):
    x=np.asarray(x,float); r=np.linalg.norm(x,axis=-1); u=x[...,2]/r
    P=MU/r*(j2*(RE/r)**2*0.5*(3*u**2-1)+j3*(RE/r)**3*0.5*(5*u**3-3*u)+j4*(RE/r)**4*(35*u**4-30*u**2+3)/8)
    return P

def energy(x,X,**kw):
    return 0.5*np.sum(X*X,-1)-MU/np.linalg.norm(x,axis=-1)+potential_P(x,**kw)

def integrate(x0,X0,ts,j2=J2,j3=J3,j4=J4,rtol=1e-13,atol=1e-12):
    f=lambda t,y: np.concatenate([y[3:],accel(y[:3],j2,j3,j4)])
    sol=solve_ivp(f,(ts[0],ts[-1]),np.concatenate([x0,X0]),method='DOP853',t_eval=ts,rtol=rtol,atol=atol)
    return sol.y[:3].T,sol.y[3:].T

def orbit(a,e,i,Om,om,f):
    """Cartesian state from osculating Keplerian elements (angles in rad)"""
    p=a*(1-e*e); r=p/(1+e*np.cos(f)); th=om+f
    cO,sO,ci,si=np.cos(Om),np.sin(Om),np.cos(i),np.sin(i)
    P=np.array([cO*np.cos(th)-sO*np.sin(th)*ci, sO*np.cos(th)+cO*np.sin(th)*ci, np.sin(th)*si])
    Q=np.array([-cO*np.sin(th)-sO*np.cos(th)*ci, -sO*np.sin(th)+cO*np.cos(th)*ci, np.cos(th)*si])
    vr=np.sqrt(MU/p)*e*np.sin(f); vt=np.sqrt(MU/p)*(1+e*np.cos(f))
    return r*P, vr*P+vt*Q

# Test orbits for v04-00 (a [km], e, i [deg]); Omega=0.3, omega=1.1, f=0.4 rad as in v03-00
CASES={'LEO':(6878.137,0.001,51.6),'SSO':(7078.137,0.001,98.2),'LEOe':(7378.137,0.05,40.0),
       'EQ':(7000.0,1e-5,0.01),'GNSS':(26560.0,0.01,55.0),'GEO':(42164.0,2e-4,0.05),
       'GTO':(24396.0,0.7283,7.0),'MOL':(26600.0,0.74,63.4),'TUN':(42164.0,0.27,63.4),
       'HEO':(70000.0,0.9,30.0)}
