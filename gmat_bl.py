"""Line-by-line Python transcription of GMAT R2026a StateConversionUtil (src/gmatutil/util/StateConversionUtil.cpp):
  BrouwerMeanShortToOsculatingElements  (lines 3681-3907)
  BrouwerMeanLongToOsculatingElements   (lines 4208-4557)
Angles in radians, a in km. Kepler solved exactly (GMAT: tol 1e-8 Short, 1e-12 Long).
The pseudostate (i > 175 deg) and negative-e branches are kept. GMAT constants are defaults but overridable
(j2..j5, re) so that J2-scaling tests can be run."""
import numpy as np
from numpy import sin as Sin, cos as Cos, sqrt as Sqrt, arctan2 as ATan2, arcsin as ASin, tan as Tan

TWO_PI = 2 * np.pi
RE = 6378.1363
J2 = 1.082626925638815E-03
J3 = -0.2532307818191774E-5
J4 = -0.1620429990000000E-5
J5 = -0.2270711043920343E-6


def Mod(x, y):
    return np.fmod(x, y)


def m2t(M, e):
    E = M + e * np.sin(M)
    for _ in range(60):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < 1e-16:
            break
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


def short(a, e, i, raan, aop, M, j2=J2, re=RE, flags=()):
    """flags: 'nolgh_sign' etc. reserved for tests"""
    smap = a / re; eccp = e; incp = i; raanp = raan; aopp = aop; meanAnom = M
    if eccp < 0.0:
        eccp = -eccp; meanAnom -= TWO_PI / 2; aopp += TWO_PI / 2
    pseudostate = 0
    if incp > np.radians(175.0):
        incp = TWO_PI / 2 - incp; raanp = -raanp; pseudostate = 1
    raanp = Mod(raanp, TWO_PI); aopp = Mod(aopp, TWO_PI); meanAnom = Mod(meanAnom, TWO_PI)
    if raanp < 0: raanp += TWO_PI
    if aopp < 0: aopp += TWO_PI
    if meanAnom < 0: meanAnom += TWO_PI
    eta = Sqrt(1.0 - eccp ** 2); theta = Cos(incp); p = smap * eta ** 2
    k2 = 0.5 * j2; gm2 = k2 / smap ** 2; gm2p = gm2 / eta ** 4
    tap = m2t(meanAnom, eccp)
    if tap < 0: tap += TWO_PI
    rp = p / (1.0 + eccp * Cos(tap)); adr = smap / rp
    sma1 = smap + smap * gm2 * ((adr ** 3 - 1.0 / eta ** 3) * (-1.0 + 3.0 * theta ** 2) + 3.0 * (1.0 - theta ** 2) * adr ** 3 * Cos(2.0 * aopp + 2.0 * tap))
    decc = eta ** 2 / 2.0 * ((3.0 * (1.0 / eta ** 6) * gm2 * (1.0 - theta ** 2) * Cos(2.0 * aopp + 2.0 * tap) * (3.0 * eccp * Cos(tap) ** 2 + 3.0 * Cos(tap) + eccp ** 2 * Cos(tap) ** 3 + eccp))
                             - gm2p * (1.0 - theta ** 2) * (3.0 * Cos(2.0 * aopp + tap) + Cos(3.0 * tap + 2.0 * aopp)) + (3.0 * theta ** 2 - 1.0) * gm2 / eta ** 6 * (eccp * eta + eccp / (1 + eta) + 3.0 * eccp * Cos(tap) ** 2
                             + 3.0 * Cos(tap) + eccp ** 2 * Cos(tap) ** 3))
    dinc = gm2p / 2.0 * theta * Sin(incp) * (3.0 * Cos(2.0 * aopp + 2.0 * tap) + 3.0 * eccp * Cos(2.0 * aopp + tap) + eccp * Cos(2.0 * aopp + 3.0 * tap))
    draan = -gm2p / 2.0 * theta * (6.0 * (tap - meanAnom + eccp * Sin(tap)) - 3.0 * Sin(2.0 * aopp + 2.0 * tap) - 3.0 * eccp * Sin(2.0 * aopp + tap) - eccp * Sin(2.0 * aopp + 3.0 * tap))
    lgh = raanp + aopp + meanAnom + gm2p / 4.0 * (6.0 * (-1.0 - 2.0 * theta + 5.0 * theta ** 2) * (tap - meanAnom + eccp * Sin(tap))
                                                 + (3.0 + 2.0 * theta - 5.0 * theta ** 2) * (3.0 * Sin(2.0 * aopp + 2.0 * tap) + 3.0 * eccp * Sin(2.0 * aopp + tap) + eccp * Sin(2.0 * aopp + 3.0 * tap))) \
        + gm2p / 4.0 * eta ** 2 / (eta + 1.0) * eccp * (3.0 * (1.0 - theta ** 2) * (Sin(3.0 * tap + 2.0 * aopp) * (1.0 / 3.0 + adr ** 2 * eta ** 2 + adr)
                                                                                   + Sin(2.0 * aopp + tap) * (1.0 - adr ** 2 * eta ** 2 - adr)) + 2.0 * Sin(tap) * (3.0 * theta ** 2 - 1.0) * (1.0 + adr ** 2 * eta ** 2 + adr))
    eccpdl = -eta ** 3 / 4.0 * gm2p * (2.0 * (-1.0 + 3.0 * theta ** 2) * (adr ** 2 * eta ** 2 + adr + 1.0) * Sin(tap)
                                       + 3.0 * (1.0 - theta ** 2) * ((-adr ** 2 * eta ** 2 - adr + 1.0) * Sin(2.0 * aopp + tap) + (adr ** 2 * eta ** 2 + adr + 1.0 / 3.0) * Sin(2.0 * aopp + 3.0 * tap)))
    ecosl = (eccp + decc) * Cos(meanAnom) - eccpdl * Sin(meanAnom)
    esinl = (eccp + decc) * Sin(meanAnom) + eccpdl * Cos(meanAnom)
    ecc1 = Sqrt(ecosl ** 2 + esinl ** 2)
    if ecc1 < 1e-11:
        ma1 = 0.0
    else:
        ma1 = ATan2(esinl, ecosl)
        if ma1 < 0: ma1 += TWO_PI
    sinhalfisinh = (Sin(0.5 * incp) + Cos(0.5 * incp) * 0.5 * dinc) * Sin(raanp) + 0.5 * Sin(incp) / Cos(incp / 2.0) * draan * Cos(raanp)
    sinhalficosh = (Sin(0.5 * incp) + Cos(0.5 * incp) * 0.5 * dinc) * Cos(raanp) - 0.5 * Sin(incp) / Cos(incp / 2.0) * draan * Sin(raanp)
    s = Sqrt(sinhalfisinh ** 2 + sinhalficosh ** 2)
    inc1 = 2.0 * ASin(min(s, 1.0))
    if inc1 == 0.0 or inc1 == 180.0:
        raan1 = 0.0
    else:
        raan1 = ATan2(sinhalfisinh, sinhalficosh)
        if raan1 < 0: raan1 += TWO_PI
    aop1 = Mod(lgh - ma1 - raan1, TWO_PI)
    if aop1 < 0: aop1 += TWO_PI
    out = [sma1 * re, ecc1, inc1, raan1, aop1, ma1]
    if pseudostate:
        out[2] = np.pi - out[2]; out[3] = TWO_PI - out[3]
    # raw corrections for white-box tests (GMAT's own variables)
    raw = dict(a=(sma1 - smap) * re, e=decc, I=dinc, el=eccpdl, h=draan, z=lgh - (raanp + aopp + meanAnom))
    return out, raw


def long(a, e, i, raan, aop, M, j2=J2, j3=J3, j4=J4, j5=J5, re=RE, crit_branch=True, t2=False):
    """t2=True: the fix proposed in the report (not GMAT): 1/(1-5cos^2 i) replaced by Phipps' T2 and no branch."""
    pseudostate = 0
    smadp = a / re; eccdp = e; incdp = i; raandp = raan; aopdp = aop; meanAnom = M
    if incdp > np.radians(175.0):
        incdp = TWO_PI / 2 - incdp; raandp = -raandp; pseudostate = 1
    raandp = Mod(raandp, TWO_PI); aopdp = Mod(aopdp, TWO_PI); meanAnom = Mod(meanAnom, TWO_PI)
    if raandp < 0: raandp += TWO_PI
    if aopdp < 0: aopdp += TWO_PI
    if meanAnom < 0: meanAnom += TWO_PI
    ae = 1.0
    bk2 = 0.5 * (j2 * ae * ae); bk3 = -j3 * ae ** 3; bk4 = -(3.0 / 8.0) * j4 * ae ** 4; bk5 = -j5 * ae ** 5
    eccdp2 = eccdp * eccdp; cn2 = 1.0 - eccdp2; cn = Sqrt(cn2)
    gm2 = bk2 / smadp ** 2; gmp2 = gm2 / (cn2 * cn2)
    gm4 = bk4 / smadp ** 4; gmp4 = gm4 / cn ** 8
    theta = Cos(incdp); theta2 = theta * theta; theta4 = theta2 * theta2
    gm3 = bk3 / smadp ** 3; gmp3 = gm3 / (cn2 * cn2 * cn2)
    gm5 = bk5 / smadp ** 5; gmp5 = gm5 / cn ** 10
    g3dg2 = gmp3 / gmp2; g4dg2 = gmp4 / gmp2; g5dg2 = gmp5 / gmp2
    sinMADP = Sin(meanAnom); cosMADP = Cos(meanAnom); sinraandp = Sin(raandp); cosraandp = Cos(raandp)
    tadp = m2t(meanAnom, eccdp)
    rp = smadp * (1.0 - eccdp * eccdp) / (1.0 + eccdp * Cos(tadp)); adr = smadp / rp
    sinta = Sin(tadp); costa = Cos(tadp); cs2gta = Cos(2.0 * aopdp + 2.0 * tadp)
    adr2 = adr * adr; adr3 = adr2 * adr; costa2 = costa * costa
    q = 1.0 - 5.0 * theta2
    if t2:      # proposed fix: every 1/q below becomes T2 = (1-exp(-100 q^2))/q (qd = 1/T2); bisubc keeps q
        T = -np.expm1(-100.0 * q * q) / q if q != 0.0 else 0.0
        qd = 1.0 / T if T != 0.0 else np.inf
    else:       # GMAT R2026a
        qd = q
    a1 = ((1.0 / 8.0) * gmp2 * cn2) * (1.0 - 11.0 * theta2 - ((40.0 * theta4) / qd))
    a2 = ((5.0 / 12.0) * g4dg2 * cn2) * (1.0 - ((8.0 * theta4) / qd) - 3.0 * theta2)
    a3 = g5dg2 * ((3.0 * eccdp2) + 4.0)
    a4 = g5dg2 * (1.0 - (24.0 * theta4) / qd - 9.0 * theta2)
    a5 = (g5dg2 * (3.0 * eccdp2 + 4.0)) * (1.0 - (24.0 * theta4) / qd - 9.0 * theta2)
    a6 = g3dg2 * (1.0 / 4.0)
    sinI = Sin(incdp); a10 = cn2 * sinI; a7 = a6 * a10
    a8p = g5dg2 * eccdp * (1.0 - (16.0 * theta4) / qd - 5.0 * theta2); a8 = a8p * eccdp
    b13 = eccdp * (a1 - a2)
    b14 = a7 + (5.0 / 64.0) * a5 * a10
    b15 = a8 * a10 * (35.0 / 384.0)
    a11 = 2.0 + eccdp2; a12 = 3.0 * eccdp2 + 2.0; a13 = theta2 * a12
    a14 = (5.0 * eccdp2 + 2.0) * (theta4 / qd)
    a17 = theta4 / (qd * qd)
    a15 = (eccdp2 * theta4 * theta2) / (qd * qd)
    a16 = theta2 / qd
    a18 = eccdp * sinI; a19 = a18 / (1.0 + cn); a21 = eccdp * theta; a22 = eccdp2 * theta
    sinI2 = Sin(incdp / 2.0); cosI2 = Cos(incdp / 2.0); tanI2 = Tan(incdp / 2.0)
    a26 = 16.0 * a16 + 40.0 * a17 + 3.0
    a27 = a22 * (1.0 / 8.0) * (11.0 + 200.0 * a17 + 80.0 * a16)
    b1 = cn * (a1 - a2) - ((a11 - 400.0 * a15 - 40.0 * a14 - 11.0 * a13) * (1.0 / 16.0) + (11.0 + 200.0 * a17 + 80.0 * a16) * a22 * (1.0 / 8.0)) * gmp2 + ((-80.0 * a15 - 8.0 * a14 - 3.0 * a13 + a11) * (5.0 / 24.0) + (5.0 / 12.0) * a26 * a22) * g4dg2
    b2 = a6 * a19 * (2.0 + cn - eccdp2) + (5.0 / 64.0) * a5 * a19 * cn2 - (15.0 / 32.0) * a4 * a18 * cn * cn2 + ((5.0 / 64.0) * a5 + a6) * a21 * tanI2 + (9.0 * eccdp2 + 26.0) * (5.0 / 64.0) * a4 * a18 + (15.0 / 32.0) * a3 * a21 * a26 * sinI * (1.0 - theta)
    b3 = ((80.0 * a17 + 5.0 + 32.0 * a16) * a22 * sinI * (theta - 1.0) * (35.0 / 576.0) * g5dg2 * eccdp) - ((a22 * tanI2 + (2.0 * eccdp2 + 3.0 * (1.0 - cn2 * cn)) * sinI) * (35.0 / 1152.0) * a8p)
    b4 = cn * eccdp * (a1 - a2)
    b5 = ((9.0 * eccdp2 + 4.0) * a10 * a4 * (5.0 / 64.0) + a7) * cn
    b6 = (35.0 / 384.0) * a8 * cn2 * cn * sinI
    if not t2:  # GMAT: (1-15th^2)/q and (1-7th^2)/q, equal to Brouwer's (1-11th^2-40th^4/q)/(1-th^2) and
                # (1-3th^2-8th^4/q)/(1-th^2), the forms used in a1 and a2, only when 1/q is not smoothed
        b7 = ((cn2 * a18) / q) * ((1.0 / 8.0) * gmp2 * (1.0 - 15.0 * theta2) + (1.0 - 7.0 * theta2) * g4dg2 * (-(5.0 / 12.0)))
    else:       # fix: Brouwer's forms with T2, so that b7 = b13 / sin i (long-period dI = -e cot i de / eta^2);
                # written with E = 1 - q T2 = exp(-100 q^2) to stay regular at sin i = 0
        E = np.exp(-100.0 * q * q)
        s2 = 1.0 - theta2
        r15 = (1.0 - 15.0 * theta2) / qd + ((1.0 - 11.0 * theta2) * E / s2 if E != 0.0 else 0.0)
        r7 = (1.0 - 7.0 * theta2) / qd + ((1.0 - 3.0 * theta2) * E / s2 if E != 0.0 else 0.0)
        b7 = (cn2 * a18) * ((1.0 / 8.0) * gmp2 * r15 + r7 * g4dg2 * (-(5.0 / 12.0)))
    b8 = (5.0 / 64.0) * (a3 * cn2 * (1.0 - 9.0 * theta2 - (24.0 * theta4 / qd))) + a6 * cn2
    b9 = a8 * (35.0 / 384.0) * cn2
    b10 = sinI * (a22 * a26 * g4dg2 * (5.0 / 12.0) - a27 * gmp2)
    b11 = a21 * (a5 * (5.0 / 64.0) + a6 + a3 * a26 * (15.0 / 32.0) * sinI * sinI)
    b12 = -((80.0 * a17 + 32.0 * a16 + 5.0) * (a22 * eccdp * sinI * sinI * (35.0 / 576.0) * g5dg2) + (a8 * a21 * (35.0 / 1152.0)))
    sma = smadp * (1.0 + gm2 * ((3.0 * theta2 - 1.0) * (eccdp2 / (cn2 * cn2 * cn2)) * (cn + (1.0 / (1.0 + cn))) + ((3.0 * theta2 - 1.0) / (cn2 * cn2 * cn2)) * (eccdp * costa) * (3.0 + 3.0 * eccdp * costa + eccdp2 * costa2) + 3.0 * (1.0 - theta2) * adr3 * cs2gta))
    sn2gta = Sin(2.0 * aopdp + 2.0 * tadp); snf2gd = Sin(2.0 * aopdp + tadp); csf2gd = Cos(2.0 * aopdp + tadp)
    sn2gd = Sin(2.0 * aopdp); cs2gd = Cos(2.0 * aopdp); sin3gd = Sin(3.0 * aopdp); cs3gd = Cos(3.0 * aopdp)
    sn3fgd = Sin(3.0 * tadp + 2.0 * aopdp); cs3fgd = Cos(3.0 * tadp + 2.0 * aopdp)
    sinGD = Sin(aopdp); cosGD = Cos(aopdp)
    bisubc = q ** -2.0 * ((25.0 * theta4 * theta) * (gmp2 * eccdp2))
    critical = bisubc >= 0.001
    if critical and crit_branch and not t2:
        dlt1e = 0.0; blghp = 0.0; eccdpdl = 0.0; dltI = 0.0; sinDH = 0.0
    else:
        blghp = raandp + aopdp + meanAnom + b3 * cs3gd + b1 * sn2gd + b2 * cosGD
        blghp = Mod(blghp, TWO_PI)
        if blghp < 0: blghp += TWO_PI
        dlt1e = b14 * sinGD + b13 * cs2gd - b15 * sin3gd
        eccdpdl = b4 * sn2gd - b5 * cosGD + b6 * cs3gd - (1.0 / 4.0) * cn2 * cn * gmp2 * (2.0 * (3.0 * theta2 - 1.0) * (adr2 * cn2 + adr + 1.0) * sinta + 3.0 * (1.0 - theta2) * ((-adr2 * cn2 - adr + 1.0) * snf2gd + (adr2 * cn2 + adr + (1.0 / 3.0)) * sn3fgd))
        dltI = 0.5 * theta * gmp2 * sinI * (eccdp * cs3fgd + 3.0 * (eccdp * csf2gd + cs2gta)) - (a21 / cn2) * (b8 * sinGD + b7 * cs2gd - b9 * sin3gd)
        sinDH = (1.0 / cosI2) * (0.5 * (b12 * cs3gd + b11 * cosGD + b10 * sn2gd - (0.5 * gmp2 * theta * sinI * (6.0 * (eccdp * sinta - meanAnom + tadp) - (3.0 * (sn2gta + eccdp * snf2gd) + eccdp * sn3fgd)))))
    blgh = blghp + ((1.0 / (cn + 1.0)) * (1.0 / 4.0) * eccdp * gmp2 * cn2 * (3.0 * (1.0 - theta2) * (sn3fgd * ((1.0 / 3.0) + adr2 * cn2 + adr) + snf2gd * (1.0 - (adr2 * cn2 + adr))) + 2.0 * sinta * (3.0 * theta2 - 1.0) * (adr2 * cn2 + adr + 1.0))) \
        + gmp2 * (3.0 / 2.0) * ((-2.0 * theta - 1.0 + 5.0 * theta2) * (eccdp * sinta + tadp - meanAnom)) + (3.0 + 2.0 * theta - 5.0 * theta2) * (gmp2 * (1.0 / 4.0) * (eccdp * sn3fgd + 3.0 * (sn2gta + eccdp * snf2gd)))
    blgh = Mod(blgh, TWO_PI)
    if blgh < 0: blgh += TWO_PI
    dlte = dlt1e + (0.5 * cn2 * ((3.0 * (1.0 / (cn2 * cn2 * cn2)) * gm2 * (1.0 - theta2) * cs2gta * (3.0 * eccdp * costa2 + 3.0 * costa + eccdp2 * costa * costa2 + eccdp))
                                 - (gmp2 * (1.0 - theta2) * (3.0 * csf2gd + cs3fgd)) + (3.0 * theta2 - 1.0) * gm2 * (1.0 / (cn2 * cn2 * cn2)) * (eccdp * cn + (eccdp / (1.0 + cn)) + 3.0 * eccdp * costa2 + 3.0 * costa + eccdp2 * costa * costa2)))
    ecc = Sqrt(eccdpdl ** 2 + (eccdp + dlte) ** 2)
    squar = (dltI * cosI2 * 0.5 + sinI2) ** 2
    sqrI = Sqrt(sinDH ** 2 + squar)
    inc = Mod(2 * ASin(sqrI), TWO_PI)
    if ecc <= 1e-11:
        aop = 0.0
        if inc <= 1e-7:
            raan_ = 0.0; ma = blgh
        else:
            arg1 = sinDH * cosraandp + sinraandp * (0.5 * dltI * cosI2 + sinI2)
            arg2 = cosraandp * (0.5 * dltI * cosI2 + sinI2) - (sinDH * sinraandp)
            raan_ = ATan2(arg1, arg2); ma = blgh - aop - raan_
    else:
        arg1 = eccdpdl * cosMADP + (eccdp + dlte) * sinMADP
        arg2 = (eccdp + dlte) * cosMADP - (eccdpdl * sinMADP)
        ma = Mod(ATan2(arg1, arg2), TWO_PI)
        if inc <= 1e-7:
            raan_ = 0.0; aop = blgh - raan_ - ma
        else:
            arg1 = sinDH * cosraandp + sinraandp * (0.5 * dltI * cosI2 + sinI2)
            arg2 = cosraandp * (0.5 * dltI * cosI2 + sinI2) - (sinDH * sinraandp)
            raan_ = ATan2(arg1, arg2); aop = blgh - ma - raan_
    if ma < 0: ma += TWO_PI
    raan_ = Mod(raan_, TWO_PI)
    if raan_ < 0: raan_ += TWO_PI
    aop = Mod(aop, TWO_PI)
    if aop < 0: aop += TWO_PI
    out = [sma * re, ecc, inc, raan_, aop, ma]
    if pseudostate:
        out[2] = np.pi - out[2]; out[3] = TWO_PI - out[3]
    return out, dict(critical=critical, bisubc=bisubc, sqrI=sqrI)
