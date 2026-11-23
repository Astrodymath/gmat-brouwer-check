"""What the three GMAT scripts give with the fix proposed in the report, without GMAT.
Usage: python fixed_values.py        (gmat_bl.py, bl_numpy.py and truth.py in the same folder; numpy, scipy)

"GMAT" lines: gmat_bl.py, the line-by-line transcription of GMAT R2026a (agrees with GMAT to 3e-7 m), so they
repeat what GmatConsole prints for the scripts. "fix" lines: the same code with the proposed fix (gmat_bl.long
with t2=True): l+g+h and the short-period terms kept, 1/(1-5cos^2 i) replaced by Phipps' T2. "check" lines: the
independent implementation bl_numpy.py with the same T2 and GMAT's constants. Part 4 integrates the
J2-J5 equations of motion (GMAT's constants) from the osculating state and compares it after one day with
the mean elements advanced at the secular rates and mapped back to osculating."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.integrate import solve_ivp
import gmat_bl as G
from bl_numpy import kep2cart, kep2equ, equ2kep, wrap_pi, mean_from_true, bl_coeffs, bl_mean_to_osc

MU = 398600.4415                               # GMAT's value in StateConversionUtil (km^3/s^2)
CONST = dict(mu=MU, re=G.RE, j2=G.J2, j3=G.J3, j4=G.J4, j5=G.J5, t2='phipps')
D, R = np.degrees, np.radians


def short(k): return G.short(*k)[0]
def long_gmat(k): return G.long(*k)[0]
def long_fix(k): return G.long(*k, t2=True)[0]


def long_check(k):                             # bl_numpy, (a, e, i, RAAN, AOP, M) in and out
    c = bl_coeffs(k[0], k[1], k[2], **CONST)
    return [float(np.ravel(v)[0]) for v in bl_mean_to_osc(c, [k[5]], [k[4]], [k[3]])]


def pos(k): return np.ravel(kep2cart(*k, mu=MU)[0])
def dr(k1, k2): return np.linalg.norm(pos(k1) - pos(k2))      # km


def show(label, k, ref=None):
    s = f"   {label:24s} a {k[0]:.6f} km, e {k[1]:.9f}, i {D(k[2]):.9f}, RAAN {D(k[3]) % 360:.9f}, " \
        f"AOP {D(k[4]) % 360:.10f}, MA {D(k[5]) % 360:.9f}"
    if ref is not None:
        d = dr(k, ref[1])
        s += f"; {ref[0]} {d:.6f} km" if d >= 1 else f"; {ref[0]} {d * 1e3:.1e} m"
    print(s)


def osc_to_mean(m2o, osc, tol=1e-13, maxit=200):
    """Fixed-point iteration on (a, ex, ey, hx, hy, true longitude), started from mean = osc."""
    ref = np.array(kep2equ(*osc), float)
    cur = ref.copy()
    for it in range(1, maxit + 1):
        d = ref - np.array(kep2equ(*m2o(list(equ2kep(*cur)))), float)
        d[5] = wrap_pi(d[5])
        cur = cur + d
        if np.max(np.abs(d) / np.array([ref[0], 1, 1, 1, 1, 1])) < tol:
            return list(equ2kep(*cur)), it
    raise RuntimeError("no convergence")


def zonal(t, y):                                # J2-J5, GMAT's constants
    x = y[:3]; r = np.linalg.norm(x); u = x[2] / r
    P = [1, u, (3 * u * u - 1) / 2, (5 * u ** 3 - 3 * u) / 2, (35 * u ** 4 - 30 * u * u + 3) / 8,
         (63 * u ** 5 - 70 * u ** 3 + 15 * u) / 8]
    dP = [0, 1, 3 * u, (15 * u * u - 3) / 2, (140 * u ** 3 - 60 * u) / 8, (315 * u ** 4 - 210 * u * u + 15) / 8]
    acc = -MU * x / r ** 3
    gu = (np.array([0.0, 0.0, 1.0]) - u * x / r) / r
    for n, J in ((2, G.J2), (3, G.J3), (4, G.J4), (5, G.J5)):
        k = MU * J * G.RE ** n / r ** (n + 1)
        acc -= -(n + 1) * k / r * P[n] * x / r + k * dP[n] * gu
    return np.concatenate([y[3:], acc])


def one_day(m2o, mean):
    """|osculating(mean advanced 1 day) - numerical integration from osculating(mean)| in km."""
    c = bl_coeffs(mean[0], mean[1], mean[2], **CONST)
    T = 86400.0
    m1 = list(mean)
    m1[3] += c.dsh * c.n0 * T; m1[4] += c.dsg * c.n0 * T; m1[5] += (1 + c.dsl) * c.n0 * T
    x0, v0 = kep2cart(*m2o(list(mean)), mu=MU)
    sol = solve_ivp(zonal, (0, T), np.concatenate([np.ravel(x0), np.ravel(v0)]), method='DOP853',
                    rtol=1e-13, atol=1e-9)
    return np.linalg.norm(sol.y[:3, -1] - pos(m2o(m1)))


print("1. BL_critical_63: mean (26600 km, 0.74, 63.4 deg, RAAN 20, AOP 270, MA 10 deg) -> osculating")
m63 = [26600.0, 0.74, R(63.4), R(20.0), R(270.0), R(10.0)]
ks = short(m63)
show("GMAT BrouwerMeanShort", ks)
show("GMAT BrouwerMeanLong", long_gmat(m63), ("distance to Short", ks))
kf = long_fix(m63)
show("fix BrouwerMeanLong", kf, ("distance to Short", ks))
show("check (bl_numpy)", long_check(m63), ("distance to fix", kf))
x = 1 - 5 * np.cos(m63[2]) ** 2
print(f"   {'':24s} (1/(1-5cos^2 i) = {1 / x:.3f}, T2 = {-np.expm1(-100 * x * x) / x:.6f})")

print("2. BL_critical_116: mean (26600 km, 0.74, 116.5 deg, RAAN 20, AOP 100, MA 50 deg) -> osculating")
m116 = [26600.0, 0.74, R(116.5), R(20.0), R(100.0), R(50.0)]
ks = short(m116)
show("GMAT BrouwerMeanShort", ks)
with np.errstate(invalid='ignore'):
    info = G.long(*m116)[1]
print(f"   GMAT BrouwerMeanLong     input to ASin = {info['sqrI']:.15g} > 1: GMAT throws")
kf = long_fix(m116)
show("fix BrouwerMeanLong", kf, ("distance to Short", ks))
show("check (bl_numpy)", long_check(m116), ("distance to fix", kf))

print("3. BL_inverse_63: osculating Keplerian (26600 km, 0.74, 63.4 deg, RAAN 20, AOP 270, TA 30 deg) -> mean")
osc = [26600.0, 0.74, R(63.4), R(20.0), R(270.0), float(mean_from_true(R(30.0), 0.74))]
ms, its = osc_to_mean(short, osc)
show("GMAT BrouwerMeanShort", ms)
print(f"   {'':24s} ({its} iterations; GmatConsole prints a 26662.34760542199, e 0.7405860341750856, i 63.40596166114632)")
print(f"   {'GMAT BrouwerMeanLong':24s} throws (see BL_inverse_63.script)")
mf, itf = osc_to_mean(long_fix, osc)
show("fix BrouwerMeanLong", mf)
print(f"   {'':24s} ({itf} iterations; round trip back to osculating: {dr(long_fix(mf), osc) * 1e3:.2e} m)")
mc, itc = osc_to_mean(long_check, osc)
show("check (bl_numpy)", mc, ("distance to fix", mf))

print("4. One day: |osculating from the mean elements advanced at the secular rates - numerical integration|")
for name, m in (("BL_critical_63 ", m63), ("BL_critical_116", m116)):
    line = f"   {name}  fix {one_day(long_fix, m):9.3f} km"
    try:
        with np.errstate(invalid='ignore'):
            g = one_day(long_gmat, m)
        line += f"   GMAT BrouwerMeanLong {g:10.3f} km" if np.isfinite(g) else "   GMAT BrouwerMeanLong throws"
    except Exception:
        line += "   GMAT BrouwerMeanLong throws"
    print(line)

print("5. The fix over a grid, 3 random angle sets per point: max |position difference|")
rng = np.random.default_rng(7)
orbits = [(7000, .001), (7000, .01), (8000, .15), (12000, .05), (20000, .5), (24400, .73), (26600, .74), (29600, .002),
          (42164, .0003)]
near = np.concatenate([np.arange(62.0, 65.0, 0.01), np.arange(115.5, 117.6, 0.01)])
far = np.concatenate([np.arange(1.0, 45.0, 0.5), np.arange(80.0, 100.0, 0.5), np.arange(140.0, 175.01, 0.5)])
res = {}
for a, e in orbits:
    for tag, grid in (("near", near), ("all", np.arange(1.0, 175.01, 0.5)), ("far", far)):
        for i in grid:
            for _ in range(3):
                k = [a, e, R(i)] + list(rng.uniform(0, 2 * np.pi, 3))
                f = long_fix(k)
                if tag == "far":
                    d = dr(f, long_gmat(k))
                else:
                    d = dr(f, long_check(k))
                res[tag] = max(res.get(tag, 0.0), d * 1e3)
print(f"   fix vs check (bl_numpy), i = 62-65 and 115.5-117.6 deg in steps of 0.01 deg: {res['near']:.1e} m")
print(f"   fix vs check (bl_numpy), i = 1-175 deg in steps of 0.5 deg:                   {res['all']:.1e} m")
print(f"   fix vs GMAT as is, i = 1-45, 80-100, 140-175 deg (far from the critical band): {res['far']:.1e} m")
print(f"   (9 orbits, a = 7000-42164 km, e = 0.0003-0.74)")
