import math

def kart2proj_fun(x, y, z, la0, m0, a, e, falseN, falseE):
    [fi, la, h] = xyz2flh(x, y, z, a, e)
    [X, Y] = FLh2GK(fi, la, la0, m0, a, e, falseN, falseE)
    return X, Y, h

def xyz2flh(x, y, z, a, e):
    d = 1
    e2 = e ** 2
    la = math.atan2(y, x)
    if x < 0:
        if y > 0:
            la = la + math.pi
        else:
            la = la - math.pi

    p = math.sqrt(x ** 2 + y ** 2)
    fi0 = math.atan(z / (p * (1 - e2)))
    N0 = a / math.sqrt(1 - e ** 2 * math.sin(fi0) ** 2)
    d_list = []

    while d > 1e-15:
        N0 = a / math.sqrt(1 - e ** 2 * math.sin(fi0) ** 2)
        hh = p / math.cos(fi0) - N0
        fi = math.atan(z / (p * (1 - e2 * N0 / (N0 + hh))))
        d = abs(fi - fi0)
        d_list.append(d)
        fi0 = fi
    d_max = max(d_list)
    d = d_max
    h = hh

    f = fi
    l = la
    return f, l, h

def FLh2GK(fi, la, la0, m0, a, e, falseN, falseE):
    l = la - la0
    Bx = DolMer(fi, a, e)
    Npol = a / math.sqrt(1 - e ** 2 * math.sin(fi) ** 2)
    t = math.tan(fi)
    ec = (e ** 2 / (1 - e ** 2)) ** 0.5
    ni = ec * math.cos(fi)

    y1 = Npol * math.cos(fi)
    y2 = Npol * ((math.cos(fi)) ** 3) * (1 - t ** 2 + ni ** 2) / 6
    y3 = Npol * ((math.cos(fi)) ** 5) * (5 - 18 * t ** 2 + t ** 4 + 14 * ni ** 2 - 58 * t ** 2 * ni ** 2) / 120

    x1 = Npol * math.sin(fi) * math.cos(fi) / 2
    x2 = Npol * math.sin(fi) * (math.cos(fi)) ** 3. * (5 - t ** 2 + 9 * ni ** 2 + 4 * ni ** 4) / 24
    x3 = Npol * math.sin(fi) * (math.cos(fi)) ** 5. * (61 - 58 * t ** 2 + t ** 4) / 720

    Y = m0 * (y1 * l + y2 * l ** 3 + y3 * l ** 5) + falseE
    X = m0 * (Bx + x1 * l ** 2 + x2 * l ** 4 + x3 * l ** 6) - falseN
    return X, Y

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

test1 = kart2proj_fun(4293741.0534,	1110007.4505, 4568994.8144, math.radians(15), 0.9999, 6378137, 0.08181919104281514, 5000000, 500000)
print(test1)


