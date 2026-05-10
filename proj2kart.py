import math
import numpy as np

def proj2kart_fun(Y, X, h, la0, m0, a, e, falseN, falseE):
    h = float(h)
    [l, f] = GK2FLh(Y, X, la0, m0, a, e, falseN, falseE)

    N = a / math.sqrt(1 - e ** 2 * math.sin(f) ** 2)
    x = (N + h) * math.cos(f) * math.cos(l)
    y = (N + h) * math.cos(f) * math.sin(l)
    z = (N * (1 - e ** 2) + h) * math.sin(f)

    return np.array((x, y, z))

def DolMer(f, a, e):
    A = 1 + 3 * e ** 2 / 4 + 45 * e ** 4 / 64 + 175 * e ** 6 / 256 + 11025 * e ** 8 / 16384 + 43659 * e ** 10 / 65536
    B = 3 * e ** 2 / 4 + 15 * e ** 4 / 16 + 525 * e ** 6 / 512 + 2205 * e ** 8 / 2048 + 72765 * e ** 10 / 65536
    C = 15 * e ** 4 / 64 + 105 * e ** 6 / 256 + 2205 * e ** 8 / 4096 + 10395 * e ** 10 / 16384
    D = 35 * e ** 6 / 512 + 315 * e ** 8 / 2048 + 31185 * e ** 10 / 131072
    E = 315 * e ** 8 / 16384 + 3465 * e ** 10 / 65536
    F = 693 * e ** 10 / 131072

    alfa = a * (1 - e ** 2) * A
    beta = a * (1 - e ** 2) * B / 2
    gama = a * (1 - e ** 2) * C / 4
    delta = a * (1 - e ** 2) * D / 6
    epsilon = a * (1 - e ** 2) * E / 8
    kapa = a * (1 - e ** 2) * F / 10

    Lm = alfa * f - beta * math.sin(2 * f) + gama * math.sin(4 * f) - delta * math.sin(6 * f) + epsilon * math.sin(8 * f) - kapa * math.sin(10 * f)

    return Lm

def GK2FLh(Y:float, X:float, la0:float, m0:float, a:float, e:float, falseN:int, falseE:int):
    Y = float(Y)
    X = float(X)
    la0 = float(la0)
    m0 = float(m0)
    a = float(a)
    e = float(e)
    falseN = int(falseN)
    falseE = int(falseE)

    yp = (Y - falseE) / m0
    xp = (X + falseN) / m0
    b = a * math.sqrt(1 - e ** 2)
    fi1 = 2 * xp / (a + b)
    Bx = DolMer(fi1, a, e)
    d = xp - Bx

    d_list = [d]

    while d > 1e-12:
        fi1 = fi1 + d / (a + b)
        Bx = DolMer(fi1, a, e)
        d = xp - Bx
        d_list.append(d)
        d_max = max(d_list)
    d = d_max

    t = math.tan(fi1)
    ec = math.sqrt(e ** 2 / (1 - e ** 2))
    ni = ec * math.cos(fi1)
    Npol = a / math.sqrt(1 - e ** 2 * math.sin(fi1) ** 2)

    f1 = -t * (1 + ni ** 2) / (2 * Npol ** 2)
    f2 = t * (5 + 3 * t ** 2 + 6 * ni ** 2 - 6 * t ** 2 * ni ** 2) / (24 * Npol ** 4)

    l1 = (Npol * math.cos(fi1)) ** -1
    l2 = -(1 + 2 * t ** 2 + ni ** 2) / (6 * Npol ** 3 * math.cos(fi1))
    l3 = (5 + 28 * t ** 2 + 24 * t ** 4) / (120 * Npol ** 5 * math.cos(fi1))

    f = fi1 + f1 * yp ** 2 + f2 * yp ** 4
    l = la0 + l1 * yp + l2 * yp ** 3 + l3 * yp ** 5

    return l, f

