"""GMAT R2026a itself (Python API, StateConversionUtil) vs the line-by-line transcription gmat_bl.py.
Usage: python real_vs_transcription.py <GMAT bin folder>   (gmat_bl.py and bl_numpy.py in the same folder)"""
import os, sys, io, contextlib
import numpy as np
BIN, HERE = sys.argv[1], os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [BIN, HERE]
import gmatpy as gmat
import gmat_bl as G
from bl_numpy import kep2cart, bl_coeffs, bl_mean_to_osc
S = gmat.StateConversionUtil
MU = 398600.4415
D = np.degrees; R = np.radians

def gm(fun, k):                       # k: a km, e, angles in rad -> GMAT result, angles in rad
    v = gmat.Rvector6(k[0], k[1], D(k[2]), D(k[3]), D(k[4]), D(k[5]))
    o = fun(MU, v)
    x = [o.GetElement(j) for j in range(6)]
    return [x[0], x[1], R(x[2]), R(x[3]), R(x[4]), R(x[5])]

def pos(k): return np.array(kep2cart(*k, mu=MU)[0]).ravel()

cases = [("LEO", 7000, 0.001, 51.6), ("LEOe", 8000, 0.15, 40), ("SSO", 7078, 0.0012, 98.2), ("GNSS", 29600, 0.002, 56),
         ("GTO", 24400, 0.73, 7), ("GEO", 42164, 0.0003, 0.05), ("HEO", 20000, 0.5, 30), ("EQ", 7200, 0.01, 0.5),
         ("TUN", 12000, 0.05, 63.0), ("MOL 63.4", 26600, 0.74, 63.4), ("MOL 63.0", 26600, 0.74, 63.0),
         ("LEOe 63.4", 8000, 0.15, 63.4), ("HEO 63.4", 20000, 0.5, 63.4)]
rng = np.random.default_rng(1)
print("A. mean -> osculating: max |position difference| over 100 random (RAAN, AOP, M), metres")
print("   theory = first-order Brouwer-Lyddane map of bl_numpy.py (raw critical divisor, GMAT constants)")
print(f"{'orbit':10s} {'Short: GMAT-transcr.':>21s} {'Long: GMAT-transcr.':>20s} {'Short: GMAT-theory':>19s} {'Long: GMAT-theory':>18s} {'Long-Short in GMAT':>19s}  branch")
for name, a, e, i in cases:
    ms = ml = mls = ts_ = tl_ = 0.0; crit = False
    cs = bl_coeffs(a, e, R(i), mu=MU, re=G.RE, j2=G.J2, j3=0, j4=0, j5=0, t2='brouwer')
    cl = bl_coeffs(a, e, R(i), mu=MU, re=G.RE, j2=G.J2, j3=G.J3, j4=G.J4, j5=G.J5, t2='brouwer')
    for _ in range(100):
        h, g, l = rng.uniform(0, 2 * np.pi, 3)
        k = [a, e, R(i), h, g, l]
        gs = gm(S.BrouwerMeanShortToOsculatingElements, k)
        with contextlib.redirect_stdout(io.StringIO()):
            gl = gm(S.BrouwerMeanLongToOsculatingElements, k)
        ts, _ = G.short(*k)
        tl, info = G.long(*k); crit |= info['critical']
        ms = max(ms, np.linalg.norm(pos(gs) - pos(ts)) * 1e3)
        ml = max(ml, np.linalg.norm(pos(gl) - pos(tl)) * 1e3)
        mls = max(mls, np.linalg.norm(pos(gl) - pos(gs)) * 1e3)
        rs = [float(np.ravel(v)[0]) for v in bl_mean_to_osc(cs, np.array([l]), np.array([g]), np.array([h]), lp=False)]
        rl = [float(np.ravel(v)[0]) for v in bl_mean_to_osc(cl, np.array([l]), np.array([g]), np.array([h]), lp=True)]
        ts_ = max(ts_, np.linalg.norm(pos(gs) - pos(rs)) * 1e3)
        tl_ = max(tl_, np.linalg.norm(pos(gl) - pos(rl)) * 1e3)
    print(f"{name:10s} {ms:21.3g} {ml:20.3g} {ts_:19.3g} {tl_:18.3g} {mls:19.3g}  {'taken' if crit else '-'}")

print()
print("B. osculating -> BrouwerMeanLong in GMAT, then back: |position difference| of the round trip, metres")
def cart(k):
    x, X = kep2cart(*k, mu=MU); return np.concatenate([np.ravel(x), np.ravel(X)])
for n, a, e, i in [("MOL 63.4", 26600, 0.74, 63.4), ("HEO 63.4", 20000, 0.5, 63.4), ("LEOe 63.4", 8000, 0.15, 63.4),
                   ("LEOe 60", 8000, 0.15, 60.0), ("MOL 55", 26600, 0.74, 55.0)]:
    osc, _ = G.long(a, e, R(i), 0.4, 4.71, 0.3, crit_branch=False)     # osculating state of the theory without the branch
    c6 = cart(osc)
    rv = gmat.Rvector6(*[float(x) for x in c6])
    buf = io.StringIO()
    try:
        mean = S.CartesianToBrouwerMeanLong(MU, rv)
        back = S.BrouwerMeanLongToCartesian(MU, mean)
        b = np.array([back.GetElement(j) for j in range(6)])
        m = [mean.GetElement(j) for j in range(6)]
        print(f"{n:10s} mean a {m[0]:10.3f} km  e {m[1]:.4f}  i {m[2]:.3f} deg   round trip |dr| {np.linalg.norm(b[:3]-c6[:3])*1e3:14.3f} m")
    except Exception as ex:
        print(f"{n:10s} exception: {" ".join(str(ex).split())}")
