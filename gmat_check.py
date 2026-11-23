"""Brouwer-Lyddane long-period conversion near the critical inclination, run in GMAT itself through its
Python API (StateConversionUtil, GMAT R2026a).
Usage: python gmat_check.py <GMAT bin folder>
bl_numpy.py (same folder) is used only for the Keplerian -> Cartesian conversion of the outputs."""
import os, sys
import numpy as np
BIN, HERE = sys.argv[1], os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [BIN, HERE]
import gmatpy as gmat
from bl_numpy import kep2cart
S = gmat.StateConversionUtil
MU, RE, J2 = 398600.4415, 6378.1363, 1.082626925638815e-3      # GMAT's values in StateConversionUtil
LONG, SHORT = S.BrouwerMeanLongToOsculatingElements, S.BrouwerMeanShortToOsculatingElements

def conv(fun, a, e, i, h, g, l):                       # km and degrees in and out
    o = fun(MU, gmat.Rvector6(a, e, i, h, g, l)); return [o.GetElement(j) for j in range(6)]
def pos(k): return np.array(kep2cart(k[0], k[1], *np.radians(k[2:]), mu=MU)[0]).ravel()
def bisubc(a, e, i):                                   # the quantity tested by the branch (>= 1e-3)
    th = np.cos(np.radians(i)); gmp2 = 0.5 * J2 / (a / RE) ** 2 / (1 - e * e) ** 2
    return 25 * th ** 5 * gmp2 * e * e / (1 - 5 * th ** 2) ** 2

print("1. Mean (26600 km, 0.74, 63.4 deg, RAAN 20, AOP 270, MA 10 deg) -> osculating")
for n, f in (("BrouwerMeanShort", SHORT), ("BrouwerMeanLong ", LONG)):
    k = conv(f, 26600.0, 0.74, 63.4, 20.0, 270.0, 10.0)
    print(f"   {n}: a {k[0]:.3f} km, e {k[1]:.6f}, i {k[2]:.5f}, RAAN {k[3]:.5f}, AOP {k[4]:.5f}, MA {k[5]:.5f} deg")

print("2. Long - Short at the same mean elements, max position difference over 200 random angles")
rng = np.random.default_rng(2)
for n, a, e, i in [("MOL 63.4", 26600, 0.74, 63.4), ("MOL 63.0", 26600, 0.74, 63.0), ("HEO 63.4", 20000, 0.5, 63.4),
                   ("LEOe 63.4", 8000, 0.15, 63.4), ("MOL 55.0", 26600, 0.74, 55.0)]:
    mx = 0.0
    for _ in range(200):
        h, g, l = np.degrees(rng.uniform(0, 2 * np.pi, 3))
        mx = max(mx, np.linalg.norm(pos(conv(LONG, a, e, i, h, g, l)) - pos(conv(SHORT, a, e, i, h, g, l))))
    print(f"   {n:10s} bisubc {bisubc(a, e, i):9.3g}   max |dr| = {mx:9.1f} km")

print("3. Scan of i in steps of 0.01 deg (3 angle sets): where |Long - Short| > 1000 km, and where Long throws")
ang = [(20.0, 100.0, 50.0), (200.0, 10.0, 300.0), (90.0, 250.0, 170.0)]
for n, a, e in [("MOL e=0.74", 26600, 0.74), ("LEOe e=0.15", 8000, 0.15), ("LEO e=0.01", 7000, 0.01)]:
    for lo, hi in ((55.0, 72.0), (108.0, 125.0)):
        big, exc, msg = [], [], ""
        for x in np.round(np.arange(lo, hi, 0.01), 2):
            mx = 0.0
            try:
                for h, g, l in ang:
                    mx = max(mx, np.linalg.norm(pos(conv(LONG, a, e, float(x), h, g, l)) - pos(conv(SHORT, a, e, float(x), h, g, l))))
            except Exception as ex:
                exc.append(x); msg = msg or " ".join(str(ex).split())[:60]; continue
            if mx > 1000: big.append(x)
        f = lambda L: f"{L[0]:.2f}-{L[-1]:.2f} deg" if L else "none"
        print(f"   {n:12s} {lo:.0f}-{hi:.0f} deg: > 1000 km at {f(big)}; exception at {f(exc)}" + (f" (e.g. {msg}...)" if msg else ""))

print("4. Cartesian -> BrouwerMeanLong (CartesianToBrouwerMeanLong) of an osculating state")
for n, a, e, i in [("MOL 63.4", 26600, 0.74, 63.4), ("HEO 63.4", 20000, 0.5, 63.4), ("LEOe 63.4", 8000, 0.15, 63.4), ("LEOe 60.0", 8000, 0.15, 60.0)]:
    k = conv(SHORT, a, e, i, np.degrees(0.4), np.degrees(4.71), np.degrees(0.3))
    x, v = kep2cart(k[0], k[1], *np.radians(k[2:]), mu=MU)
    try:
        m = S.CartesianToBrouwerMeanLong(MU, gmat.Rvector6(*[float(c) for c in np.concatenate([np.ravel(x), np.ravel(v)])]))
        print(f"   {n:10s} converges: mean a {m.GetElement(0):.3f} km, e {m.GetElement(1):.4f}, i {m.GetElement(2):.3f} deg")
    except Exception as ex:
        print(f"   {n:10s} " + " ".join(str(ex).split()))
