"""Brouwer-Lyddane (BL) analytic propagator for the zonal problem J2, J3, J4 (J5 optional, default 0).

Transcribed from the Orekit BrouwerLyddanePropagator.BLModel (Apache-2.0, develop branch 2026, the
"Lyddane-variable" form introduced in Orekit 12.2), which follows
  Brouwer 1959 (AJ 64, 378), Lyddane 1963 (AJ 68, 555), Phipps 1992 (NPS thesis) / Solomon 1991 (NAVSPASUR),
with five coefficients corrected (variant='corrected', default; variant='orekit' = Orekit 12.2+ literal):
  short period (J2):
  * delta2 e : 3(1-th^2)(w20 + e) cos(2g+2f)      [Orekit 12.2+: 3(1-th^2)(w20 + e cos(2g+2f))]
  * delta2 e : -eta^2 (1-th^2)(3cos(2g+f)+cos(2g+3f)) [Orekit 12.2+: extra factor e]
  * e delta2 l: 2(3th^2-1)(w22+1) sin f           [Orekit 12.2+: 2(1-th^2)(w22+1) sin f]
  long period (from conservation of H: dI = -e cotI de / eta^2):
  * vli1 = -e^2 cosI q1 / sinI                    [Orekit 12.2+: -e q1 / sinI]     (J2^2, J4)
  * vli3 = +3 e^2 cosI p4                         [Orekit 12.2+: -3 e^2 cosI p4]   (J5 only)
The corrected short-period terms agree with Brouwer 1959 / Orekit <= 12.1 and with an independent sympy
derivation from Brouwer's generating function; the corrected long-period inclination terms agree with
the long-period generating function; both are confirmed by the numerical averaging test (validate_bl.py).

Content of the theory (as in Orekit, M2 drag term = 0):
  * secular rates of l, g, h: second order in J2, first order in J4 (J3, J5 have none);
  * long-period terms: J2^2/J2, J3/J2, J4/J2, J5/J2 (first order), critical-inclination factor
    1/(1-5cos^2 I) replaced by Phipps' T2 = (1-exp(-100 x^2))/x  (t2='phipps', Orekit default) or kept raw
    (t2='brouwer');
  * short-period terms: first order in J2 only;
  * Lyddane's non-singular assembly (e'' + de, e''dl, sin(I''/2)+cos(I''/2)dI/2, sin(I''/2)dh, l+g+h);
  * osculating -> mean: fixed-point iteration on equinoctial elements (a, ex, ey, hx, hy, L_true), as in
    Orekit FixedPointConverter (damping 1, threshold 1e-13, max 200 iterations), started from mean = osc.

Units: km, s, rad.  State output: osculating Cartesian position/velocity from the osculating Keplerian
elements (velocity = two-body velocity of the osculating elements, which is exact for osculating elements).
"""
import os, sys, warnings
import numpy as np

_V4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _V4 not in sys.path:
    sys.path.insert(0, _V4)
from truth import MU, RE, J2, J3, J4  # noqa: E402

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------------------------------
# two-body utilities (vectorised)
# ----------------------------------------------------------------------------------------------
def wrap_pi(x):
    return (np.asarray(x) + np.pi) % TWO_PI - np.pi


def kepler_E(M, e, tol=1e-15, maxit=60):
    """Eccentric anomaly from mean anomaly, elliptic, vectorised Newton with Danby's starter."""
    M = np.asarray(M, float)
    e = np.asarray(e, float)
    Mr = wrap_pi(M)
    E = Mr + 0.85 * e * np.sign(np.sin(Mr) + (Mr == 0))
    for _ in range(maxit):
        se, ce = np.sin(E), np.cos(E)
        f0 = E - e * se - Mr
        f1 = 1.0 - e * ce
        f2 = e * se
        d = -f0 / f1
        d = -f0 / (f1 + 0.5 * d * f2)          # Halley
        E = E + d
        if np.all(np.abs(d) < tol):
            break
    return E + (M - Mr)


def true_from_mean(M, e):
    E = kepler_E(M, e)
    eta = np.sqrt(1.0 - e * e)
    f = np.arctan2(eta * np.sin(E), np.cos(E) - e)
    # keep the same revolution count as M
    return f + (np.round((E - f) / TWO_PI) * TWO_PI), E


def mean_from_true(f, e):
    eta = np.sqrt(1.0 - e * e)
    E = np.arctan2(eta * np.sin(f), e + np.cos(f))
    return E - e * np.sin(E)


def kep2cart(a, e, i, Om, om, M, mu=MU):
    """Cartesian state from Keplerian elements (mean anomaly). Broadcasts; returns (...,3) arrays."""
    a, e, i, Om, om, M = np.broadcast_arrays(*[np.asarray(v, float) for v in (a, e, i, Om, om, M)])
    E = kepler_E(M, e)
    cE, sE = np.cos(E), np.sin(E)
    eta = np.sqrt(1.0 - e * e)
    n = np.sqrt(mu / a ** 3)
    xp = a * (cE - e)
    yp = a * eta * sE
    den = 1.0 - e * cE
    vxp = -a * n * sE / den
    vyp = a * n * eta * cE / den
    cO, sO, ci, si, co, so = np.cos(Om), np.sin(Om), np.cos(i), np.sin(i), np.cos(om), np.sin(om)
    P = np.stack([cO * co - sO * so * ci, sO * co + cO * so * ci, so * si], -1)
    Q = np.stack([-cO * so - sO * co * ci, -sO * so + cO * co * ci, co * si], -1)
    x = xp[..., None] * P + yp[..., None] * Q
    X = vxp[..., None] * P + vyp[..., None] * Q
    return x, X


def cart2kep(x, X, mu=MU):
    """Osculating Keplerian elements (a, e, i, Om, om, M) from Cartesian state (...,3)."""
    x = np.asarray(x, float)
    X = np.asarray(X, float)
    r = np.linalg.norm(x, axis=-1)
    v2 = np.sum(X * X, -1)
    h = np.cross(x, X)
    hn = np.linalg.norm(h, axis=-1)
    a = 1.0 / (2.0 / r - v2 / mu)
    ev = np.cross(X, h) / mu - x / r[..., None]
    e = np.linalg.norm(ev, axis=-1)
    i = np.arctan2(np.hypot(h[..., 0], h[..., 1]), h[..., 2])
    Om = np.arctan2(h[..., 0], -h[..., 1])
    P = np.stack([np.cos(Om), np.sin(Om), np.zeros_like(Om)], -1)
    Q = np.cross(h / hn[..., None], P)
    u = np.arctan2(np.sum(x * Q, -1), np.sum(x * P, -1))
    om = np.arctan2(np.sum(ev * Q, -1), np.sum(ev * P, -1))
    f = u - om
    M = mean_from_true(f, e)
    return a, e, i, Om % TWO_PI, om % TWO_PI, M % TWO_PI


def kep2equ(a, e, i, Om, om, M):
    """(a, ex, ey, hx, hy, Lv) with hx=tan(i/2)cosOm, Lv = Om+om+f (true longitude)."""
    f, _ = true_from_mean(M, e)
    t = np.tan(0.5 * i)
    return (a, e * np.cos(om + Om), e * np.sin(om + Om), t * np.cos(Om), t * np.sin(Om), Om + om + f)


def equ2kep(a, ex, ey, hx, hy, Lv):
    e = np.hypot(ex, ey)
    i = 2.0 * np.arctan(np.hypot(hx, hy))
    Om = np.arctan2(hy, hx)
    pa = np.arctan2(ey, ex)          # om + Om
    om = pa - Om
    f = Lv - pa
    M = mean_from_true(f, e)
    return a, e, i, Om, om, M


# ----------------------------------------------------------------------------------------------
# Brouwer-Lyddane model
# ----------------------------------------------------------------------------------------------
def T2_phipps(ci):
    """Orekit/Phipps smooth replacement of 1/(1-5cos^2 I): (1-exp(-100 x^2))/x, x = 1-5cos^2 I.
    (Orekit evaluates the same function through a 13-term series and an 11-factor product,
    beta = 100/2048; the two agree to ~1e-13, see validate_bl.py.)"""
    x = 1.0 - 5.0 * ci * ci
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -np.expm1(-100.0 * x * x) / x
    return np.where(x == 0.0, 0.0, t)


def T2_orekit(ci):
    """Literal transcription of Orekit's T2 (Phipps 1992 eq. 2.47-2.48), for validation."""
    from math import factorial
    beta = 100.0 / 2048.0
    x = 1.0 - 5.0 * ci * ci
    x2 = x * x
    s = sum((-1) ** k * beta ** k * x2 ** k / factorial(k + 1) for k in range(13))
    p = 1.0
    for k in range(11):
        p *= 1.0 + np.exp(-(2.0 ** k) * beta * x2)
    return beta * x * s * p


class _Coef:
    pass


def bl_coeffs(app, epp, ipp, mu=MU, re=RE, j2=J2, j3=J3, j4=J4, j5=0.0, t2='phipps'):
    """Constants of the BL solution for mean (a'', e'', I''). Transcription of Orekit BLModel(...)."""
    if not (0.0 <= epp < 1.0):
        raise ValueError('BL needs 0 <= e" < 1, got %r' % epp)
    ck = [0.0, 0.0, -j2, -j3, -j4, -j5]          # C_n0 = -J_n
    c = _Coef()
    c.app, c.epp, c.ipp = app, epp, ipp
    epp2 = epp * epp
    n2 = 1.0 - epp2
    n = np.sqrt(n2)
    n3 = n2 * n
    t8 = n + 1.0 / (1.0 + n)
    n0 = np.sqrt(mu / app) / app
    q = re / app
    ql = q * q
    nl = n2 * n2
    yp2 = -0.5 * ck[2] * ql / nl                    # gamma2' = J2/2 (Re/a)^2 / eta^4
    yp22 = yp2 * yp2
    ql *= q; nl *= n2
    yp3 = ck[3] * ql / nl                           # gamma3' = -J3 (Re/a)^3 / eta^6
    ql *= q; nl *= n2
    yp4 = 0.375 * ck[4] * ql / nl                   # gamma4' = -3/8 J4 (Re/a)^4 / eta^8
    ql *= q; nl *= n2
    yp5 = ck[5] * ql / nl                           # gamma5' = -J5 (Re/a)^5 / eta^10
    si, ci = np.sin(ipp), np.cos(ipp)
    ci2 = ci * ci
    oneMci2 = 1.0 - ci2
    ci2X3M1 = 3.0 * ci2 - 1.0
    ci2X5M1 = 5.0 * ci2 - 1.0

    # secular rates (in units of n0):  l = l0 + (1+dsl) n0 t, g = g0 + dsg n0 t, h = h0 + dsh n0 t
    dsl = 1.5 * yp2 * n * (ci2X3M1 +
                           0.0625 * yp2 * (-15.0 + n * (16.0 + 25.0 * n) +
                                           ci2 * (30.0 - n * (96.0 + 90.0 * n) +
                                                  ci2 * (105.0 + n * (144.0 + 25.0 * n))))) + \
        0.9375 * yp4 * n * epp2 * (3.0 - ci2 * (30.0 - 35.0 * ci2))
    dsg = 1.5 * yp2 * ci2X5M1 + \
        0.09375 * yp22 * (-35.0 + n * (24.0 + 25.0 * n) +
                          ci2 * (90.0 - n * (192.0 + 126.0 * n) +
                                 ci2 * (385.0 + n * (360.0 + 45.0 * n)))) + \
        0.3125 * yp4 * (21.0 - 9.0 * n2 + ci2 * (-270.0 + 126.0 * n2 + ci2 * (385.0 - 189.0 * n2)))
    dsh = (-3.0 * yp2 +
           0.375 * yp22 * (-5.0 + n * (12.0 + 9.0 * n) - ci2 * (35.0 + n * (36.0 + 5.0 * n))) +
           1.25 * yp4 * (5.0 - 3.0 * n2) * (3.0 - 7.0 * ci2)) * ci

    # critical-inclination factor 1/(1-5 cos^2 I)
    if t2 == 'phipps':
        tt2 = T2_phipps(ci)
    elif t2 == 'brouwer':
        tt2 = 1.0 / (1.0 - 5.0 * ci2)
    else:
        raise ValueError(t2)

    # long-period factors (yp2 != 0 required)
    fs12 = yp3 / yp2
    fs13 = 10.0 * yp4 / (3.0 * yp2)
    fs14 = yp5 / yp2
    ci2Xt2 = ci2 * tt2
    cA = 1.0 - ci2 * (11.0 + 40.0 * ci2Xt2)
    cB = 1.0 - ci2 * (3.0 + 8.0 * ci2Xt2)
    cC = 1.0 - ci2 * (9.0 + 24.0 * ci2Xt2)
    cD = 1.0 - ci2 * (5.0 + 16.0 * ci2Xt2)
    cE = 1.0 - ci2 * (33.0 + 200.0 * ci2Xt2)
    cF = 1.0 - ci2 * (9.0 + 40.0 * ci2Xt2)
    p5p = 1.0 + ci2Xt2 * (8.0 + 20.0 * ci2Xt2)
    p5p2 = 1.0 + 2.0 * p5p
    p5p4 = 1.0 + 4.0 * p5p
    p5p10 = 1.0 + 10.0 * p5p
    e2X3P4 = 4.0 + 3.0 * epp2
    ciO1Pci = ci / (1.0 + ci)
    q1 = 0.125 * (yp2 * cA - fs13 * cB)
    q2 = 0.125 * epp2 * ci * (yp2 * p5p10 - fs13 * p5p2)
    q5 = 0.25 * (fs12 + 0.3125 * e2X3P4 * fs14 * cC)
    p2 = 0.46875 * p5p2 * epp * ci * si * e2X3P4 * fs14
    p3 = 0.15625 * epp * si * fs14 * cC
    kf = 35.0 / 1152.0
    p4 = kf * epp * fs14 * cD
    p5 = 2.0 * kf * epp * epp2 * ci * si * fs14 * p5p4

    c.vle1 = epp * n2 * q1
    c.vle2 = n2 * si * q5
    c.vle3 = -3.0 * epp * n2 * si * p4
    # Orekit 12.2+ literal (with Orekit's guard: dropped when not finite, i.e. sin I'' = 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        c.vli1_orekit = -epp * q1 / si
    c.vli3_orekit = -3.0 * epp2 * ci * p4
    # corrected: dI = -(e cotI / eta^2) de  term by term (H is not changed by the long-period transform);
    # q1 ~ sin^2 I near I = 0, so vli1 -> 0 there
    c.vli1 = -epp2 * ci * q1 / si if si != 0.0 else 0.0
    c.vli2 = -epp * ci * q5
    c.vli3 = 3.0 * epp2 * ci * p4
    c.vll2 = c.vle2 + 3.0 * epp * n2 * p3
    c.vlh1I = -si * q2
    c.vlh2I = epp * ci * q5 + si * p2
    c.vlh3I = -epp2 * ci * p4 - si * p5
    c.vls1 = (n3 - 1.0) * q1 - q2 + 25.0 * epp2 * ci2 * ci2Xt2 * ci2Xt2 * (yp2 - 0.2 * fs13) - \
        0.0625 * epp2 * (yp2 * cE - fs13 * cF)
    c.vls2 = epp * si * (t8 + ciO1Pci) * q5 + (11.0 + 3.0 * (epp2 - n3)) * p3 + (1.0 - ci) * p2
    c.vls3 = si * p4 * (3.0 * (n3 - 1.0) - epp2 * (2.0 + ciO1Pci)) - (1.0 - ci) * p5

    for k, v in dict(n0=n0, n=n, n2=n2, n3=n3, t8=t8, yp2=yp2, yp3=yp3, yp4=yp4, yp5=yp5, si=si, ci=ci,
                     ci2=ci2, oneMci2=oneMci2, ci2X3M1=ci2X3M1, dsl=dsl, dsg=dsg, dsh=dsh, tt2=tt2,
                     q1=q1, q2=q2, q5=q5, siO2=np.sin(0.5 * ipp), ciO2=np.cos(0.5 * ipp)).items():
        setattr(c, k, v)
    return c


def bl_mean_to_osc(c, lpp, gpp, hpp, lp=True, sp=True, variant='corrected'):
    """Osculating Keplerian elements (a, e, i, Om, om, M) from mean angles l'', g'', h'' (arrays)
    for the constants c = bl_coeffs(a'', e'', I''). Transcription of Orekit BLModel.propagateParameters.
    lp/sp switch long-/short-period terms off (for tests); variant see module docstring."""
    if variant not in ('corrected', 'orekit'):
        raise ValueError(variant)
    ok = variant == 'corrected'
    lpp = np.asarray(lpp, float)
    gpp = np.asarray(gpp, float)
    hpp = np.asarray(hpp, float)
    app, epp, n, n2, n3, t8 = c.app, c.epp, c.n, c.n2, c.n3, c.t8
    yp2, ci, si, oneMci2, ci2X3M1 = c.yp2, c.ci, c.si, c.oneMci2, c.ci2X3M1

    # ---- long-period corrections
    cg, sg = np.cos(gpp), np.sin(gpp)
    c2g, s2g = np.cos(2 * gpp), np.sin(2 * gpp)
    c3g, s3g = np.cos(3 * gpp), np.sin(3 * gpp)
    if lp:
        d1e = c2g * c.vle1 + sg * c.vle2 + s3g * c.vle3
        if ok:
            d1I = sg * c.vli2 + s3g * c.vli3 + c2g * c.vli1
        else:
            d1I = sg * c.vli2 + s3g * c.vli3_orekit
            if np.isfinite(c.vli1_orekit):
                d1I = d1I + c2g * c.vli1_orekit
        eppd1l = (s2g * c.vle1 - cg * c.vll2 - c3g * c.vle3) * n
        sIppd1h = s2g * c.vlh1I + cg * c.vlh2I + c3g * c.vlh3I
        d1z = s2g * c.vls1 + cg * c.vls2 + c3g * c.vls3
    else:
        d1e = d1I = eppd1l = sIppd1h = d1z = 0.0

    # ---- short-period corrections (first order in J2)
    fpp, _ = true_from_mean(lpp, epp)
    cf, sf = np.cos(fpp), np.sin(fpp)
    if sp:
        esf = epp * sf
        ecf = epp * cf
        P1 = ecf + 1.0
        P2 = ecf + 2.0
        P3 = ecf + 3.0
        P1_3 = P1 * P1 * P1
        g2 = 2.0 * gpp
        c2gf, s2gf = np.cos(g2 + fpp), np.sin(g2 + fpp)
        c2g2f, s2g2f = np.cos(g2 + 2 * fpp), np.sin(g2 + 2 * fpp)
        c2g3f, s2g3f = np.cos(g2 + 3 * fpp), np.sin(g2 + 3 * fpp)
        ec2gf, es2gf = epp * c2gf, epp * s2gf
        ec2g3f, es2g3f = epp * c2g3f, epp * s2g3f
        w17 = fpp + esf - lpp                               # f + e sin f - l
        w20 = cf * (P3 * ecf + 3.0)                          # ((e cf + 3) e cf + 3) cf
        w21 = 3.0 * (s2g2f + es2gf) + es2g3f                 # 3 sin(2g+2f) + 3e sin(2g+f) + e sin(2g+3f)
        w22 = P1 * P2 / n2                                   # a^2 eta^2/r^2 + a/r
        d2a = app * (yp2 / n2) * ((P1_3 - n3) * ci2X3M1 + 3.0 * oneMci2 * P1_3 * c2g2f)
        if ok:
            d2e = 0.5 * yp2 * ((w20 + epp * t8) * ci2X3M1 + 3.0 * oneMci2 * (w20 + epp) * c2g2f
                               - n2 * oneMci2 * (3.0 * c2gf + c2g3f))
            k_sf = ci2X3M1
        else:  # Orekit >= 12.2 literal
            d2e = 0.5 * yp2 * ((w20 + epp * t8) * ci2X3M1 + 3.0 * oneMci2 * (w20 + epp * c2g2f)
                               - n2 * oneMci2 * (3.0 * ec2gf + ec2g3f))
            k_sf = oneMci2
        d2I = 0.5 * yp2 * ci * si * (3.0 * (c2g2f + ec2gf) + ec2g3f)
        eppd2l = -0.25 * yp2 * n3 * (2.0 * k_sf * (w22 + 1.0) * sf +
                                     3.0 * oneMci2 * (-(w22 - 1.0) * s2gf + (w22 + 1.0 / 3.0) * s2g3f))
        sIppd2h = 0.5 * yp2 * ci * si * (w21 - 6.0 * w17)
        d2z = -(epp * eppd2l * (t8 - 1.0) / n3 +
                0.25 * yp2 * (6.0 * (1.0 + ci * (2.0 - 5.0 * ci)) * w17 - (3.0 + ci * (2.0 - 5.0 * ci)) * w21))
    else:
        d2a = d2e = d2I = eppd2l = sIppd2h = d2z = 0.0

    # ---- Lyddane assembly
    de = epp + d1e + d2e
    dl = eppd1l + eppd2l
    dh = (sIppd1h + sIppd2h) / (2.0 * c.ciO2)
    di = 0.5 * c.ciO2 * (d1I + d2I) + c.siO2
    z = lpp + gpp + hpp + d1z + d2z
    a = app + d2a
    e = np.hypot(de, dl)
    clpp, slpp = np.cos(lpp), np.sin(lpp)
    l = np.arctan2(de * slpp + dl * clpp, de * clpp - dl * slpp)
    i = 2.0 * np.arcsin(np.minimum(np.hypot(di, dh), 1.0))
    chpp, shpp = np.cos(hpp), np.sin(hpp)
    h = np.arctan2(di * shpp + dh * chpp, di * chpp - dh * shpp)
    g = z - l - h
    a, e, i, h, g, l = np.broadcast_arrays(a, e, i, h, g, l)
    return a, e, i, h, g, l


class BL:
    """Brouwer-Lyddane propagator.

    BL(x0, X0, t0=0, j2=J2, j3=J3, j4=J4, j5=0, mu=MU, re=RE, mean=False, t2='phipps',
       eps=1e-13, maxit=200, variant='corrected')
      x0, X0 : osculating position/velocity at t0 [km, km/s] (or mean, with mean=True: then x0, X0 are
               the Cartesian image of the *mean* Keplerian elements)
    .state(t) -> x (N,3), X (N,3)  osculating Cartesian state at times t [s]
    .osc_elements(t) -> (a, e, i, Om, om, M) osculating Keplerian
    .mean_elements(t) -> (a'', e'', I'', h'', g'', l'')
    Attributes: mean0 (a'',e'',I'',h'',g'',l'') at t0, niter, resid (fixed-point iteration info).
    """

    def __init__(self, x0, X0, t0=0.0, j2=J2, j3=J3, j4=J4, j5=0.0, mu=MU, re=RE, mean=False,
                 t2='phipps', eps=1e-13, maxit=200, variant='corrected', lp=True, sp=True):
        self.t0 = float(t0)
        self.kw = dict(mu=mu, re=re, j2=j2, j3=j3, j4=j4, j5=j5, t2=t2)
        self.opt = dict(lp=lp, sp=sp, variant=variant)
        self.mu = mu
        osc = cart2kep(np.asarray(x0, float), np.asarray(X0, float), mu)
        self.osc0 = tuple(float(v) for v in osc)
        if mean:
            a, e, i, Om, om, M = self.osc0
            self.mean0 = (a, e, i, Om, om, M)
            self.niter, self.resid, self.converged = 0, 0.0, True
        else:
            self.mean0, self.niter, self.resid, self.converged = self._osc2mean(self.osc0, eps, maxit)
        a, e, i, h, g, l = self.mean0
        self.c = bl_coeffs(a, e, i, **self.kw)

    # mean -> osculating at the same epoch (dt = 0), Keplerian in / Keplerian out
    def _m2o(self, mk):
        a, e, i, Om, om, M = mk
        c = bl_coeffs(a, e, i, **self.kw)
        return bl_mean_to_osc(c, M, om, Om, **self.opt)

    def _osc2mean(self, osc, eps, maxit):
        """Orekit FixedPointConverter: iterate on equinoctial (a, ex, ey, hx, hy, Lv), damping 1."""
        ref = np.array(kep2equ(*osc), float)
        cur = ref.copy()
        thr = eps * np.array([abs(ref[0]), 1 + np.hypot(ref[1], ref[2]), 1 + np.hypot(ref[1], ref[2]),
                              1 + np.hypot(ref[3], ref[4]), 1 + np.hypot(ref[3], ref[4]), np.pi])
        resid = np.inf
        for it in range(1, maxit + 1):
            mk = equ2kep(*cur)
            try:
                upd = np.array(kep2equ(*[float(v) for v in self._m2o(mk)]), float)
            except ValueError as ex:
                raise RuntimeError('BL osc->mean fixed point diverged at iteration %d (%s)' % (it, ex)) from None
            if not np.all(np.isfinite(upd)):
                raise RuntimeError('BL osc->mean fixed point diverged at iteration %d (non-finite)' % it)
            d = ref - upd
            d[5] = wrap_pi(d[5])
            resid = float(np.max(np.abs(d) / thr * eps))
            if np.all(np.abs(d) < thr):
                mk = equ2kep(*cur)
                return tuple(float(v) for v in mk), it, resid, True
            cur = cur + d
        warnings.warn('BL osc->mean did not converge in %d iterations (max scaled residual %.3e)' % (maxit, resid))
        mk = equ2kep(*cur)
        return tuple(float(v) for v in mk), maxit, resid, False

    def mean_elements(self, t):
        dt = np.asarray(t, float) - self.t0
        a, e, i, h, g, l = self.mean0
        c = self.c
        nt = c.n0 * dt
        return (a, e, i, h + c.dsh * nt, g + c.dsg * nt, l + (1.0 + c.dsl) * nt)

    def osc_elements(self, t):
        a, e, i, h, g, l = self.mean_elements(t)
        a_, e_, i_, h_, g_, l_ = bl_mean_to_osc(self.c, l, g, h, **self.opt)
        return a_, e_, i_, h_, g_, l_

    def state(self, t):
        a, e, i, Om, om, M = self.osc_elements(np.atleast_1d(np.asarray(t, float)))
        return kep2cart(a, e, i, Om, om, M, self.mu)


def osc_to_mean(x, X, **kw):
    """Mean BL elements (a'', e'', I'', h'', g'', l'') for one Cartesian state."""
    return BL(x, X, **kw).mean0
