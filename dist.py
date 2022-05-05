from numpy import pi, cos, sin, arccos, arcsin, dot
import numpy as np
import random as rn 

# ">> size >> steps >> box_size >> time_step >> cut >> <fn>"
# [x, y, z, vx, vy, vz, mass]

# G = 1.5607939e-22 # klyr3 / (solMass yr2)
G = 1.5607939e-13 # lyr3 / (solMass yr2)

def normal(phi, psi):
    v1 = np.array([cos(phi), 0, sin(phi)])
    v2 = np.array([0, cos(psi), sin(psi)])
    return np.cross(v1, v2)

def disk(
        size, box_size,
        x0, y0, z0, disk_mass, main_mass,
        vx0=0., vy0=0., vz0=0.,
        phi0=0.0, psi0=0.0):
    points = []

    n = a, b, c = normal(phi0, psi0)

    mass = disk_mass / size
    dens = size / (box_size * box_size)
    r_list = []
    for i in range(size):
        r = rn.uniform(0.0, box_size) + 4e3
        thet = rn.uniform(0.0000001, 2.0 * pi) #+ phi0

        i = r * cos(thet)
        j = r * sin(thet)

        x = i * cos(phi0)
        y = j * cos(psi0)
        #z = np.sqrt((sin(phi0) ** 2 * i ** 2) + (sin(psi0) ** 2 * j ** 2))
        z = (-(a * x) - (b * y)) / c

        m0 = 0 #len(np.where(np.array(r_list) < r)[0]) * mass
        v = 1 * np.sqrt(((G * (main_mass + m0)) / r))  # / np.sqrt(2)

        vi = v * -sin(thet)
        vj = v * cos(thet)

        vx = vi * cos(phi0)
        vy = vj * cos(psi0)
        #vz = np.sqrt((sin(phi0) ** 2 * vi ** 2) + (sin(psi0) ** 2 * vj ** 2))
        vz = (-(a * vx) - (b * vy)) / c

        line = [str(j) for j in [x+x0, y+y0, z+z0, vx+vx0, vy+vy0, vz+vz0, mass]]
        points.append(line)
        r_list.append(r)

    points[0] = [str(j) for j in [x0, y0, z0, vx0, vy0, vz0, main_mass]]
    return points


def uniform_disk(
        size, box_size,
        x0, y0, z0, disk_mass, main_mass,
        vx0=0., vy0=0., vz0=0.,
        phi0=0.0, psi0=0.0):
    points = []

    n = a, b, c = normal(phi0, psi0)

    mass = disk_mass / size
    dens = size / (box_size * box_size)
    r_list = []
    for i in range(size):
        r_max = box_size + 4e3
        i_lim = r_max
        r = np.inf
        while r > r_max:
            i = rn.uniform(-i_lim, i_lim)

            j_lim = i_lim#np.sqrt((i_lim)**2 - i**2)

            j = rn.uniform(-j_lim, j_lim)

            r = np.sqrt(i**2 + j**2)
            thet = np.arcsin(j/r) % (2*np.pi)

        x = i * cos(phi0)
        y = j * cos(psi0)
        #z = np.sqrt((sin(phi0) ** 2 * i ** 2) + (sin(psi0) ** 2 * j ** 2))
        z = (-(a * x) - (b * y)) / c

        m0 = 0 #len(np.where(np.array(r_list) < r)[0]) * mass
        v = 1 * np.sqrt(((G * (main_mass + m0)) / r))  # / np.sqrt(2)

        vi = v * -sin(thet)
        vj = v * cos(thet) if x > 0 else - v * cos(thet)

        vx = vi * cos(phi0)
        vy = vj * cos(psi0)
        #vz = np.sqrt((sin(phi0) ** 2 * vi ** 2) + (sin(psi0) ** 2 * vj ** 2))
        vz = (-(a * vx) - (b * vy)) / c

        line = [str(j) for j in [x+x0, y+y0, z+z0, vx+vx0, vy+vy0, vz+vz0, mass]]
        points.append(line)
        r_list.append(r)

    points[0] = [str(j) for j in [x0, y0, z0, vx0, vy0, vz0, main_mass]]
    return points

def stable_sphere(size, box_size,
                  x, y, z, m, main_mass,
                  vx=0., vy=0., vz=0.):
    points = []
    if size == 0:
        return points

    for i in range(size):
        
        r = rn.uniform(0.0, box_size)
        phi = rn.uniform(0.0000001, 2.0 * pi)
        costheta = rn.uniform(-1,1)
        thet = arccos(costheta)

        xi = r*sin(thet)*cos(phi)
        yi = r*costheta
        zi = r*sin(thet)*sin(phi)
                
        phi = rn.uniform(0.0000001, 2.0 * pi)
        costheta = rn.uniform(-1,1)
        thet = arccos(costheta)

        v = ( G*(main_mass) / (r+1) )**(0.5)

        vxi = v*sin(thet)*cos(phi)
        vyi = v*costheta
        vzi = v*sin(thet)*sin(phi)

        line = [str(j) for j in [x+xi,y+yi,z+zi,vxi+vx,vyi+vy,vzi+vz, m]]
        points.append(line)
    
    points[0] = [str(j) for j in [x, y, z, vx, vy, vz, main_mass]]
    return points

def collapse_shpere(size, box_size,
                    x, y, z, m,
                    vx=0., vy=0., vz=0.):
    points = []

    for i in range(size):
        r = rn.uniform(0.0, box_size)
        phi = rn.uniform(0.0000001, 2.0 * pi)
        costheta = rn.uniform(-1,1)
        thet = arccos(costheta)

        xi = r*sin(thet)*cos(phi)
        yi = r*costheta
        zi = r*sin(thet)*sin(phi)

        vxi = 0.
        vyi = 0.
        vzi = 0.

        line = [str(j) for j in [x+xi,y+yi,z+zi,vxi+vx,vyi+vy,vzi+vz, m]]
        points.append(line)
    return points


def scalar_to_sph(r):
    phi = rn.uniform(0.0000001, 2.0 * pi)

    costheta = rn.uniform(-1, 1)
    thet = arccos(costheta)

    x = r * sin(thet) * cos(phi)
    y = r * sin(thet) * sin(phi)
    z = r * costheta

    return x, y, z, phi, thet


def plummer(size, box_size,
            x0, y0, z0, total_mass,
            vx0=0., vy0=0., vz0=0.):

    a = box_size
    M = total_mass

    points = []
    if size == 0:
        return points

    mass = M/size

    for i in range(size):
        uni_draw = np.random.uniform(0, 1)
        r = a * (uni_draw**(-2/3) - 1)**(-1/2)

        x, y, z, *_ = scalar_to_sph(r)

        # Von Neumann acceptance-rejection technique
        q = 0
        g_of_q = 0.1
        while g_of_q < q**2 * (1 - q**2)**3.5:
            q = np.random.uniform(0.5, 1)
            g_of_q = np.random.uniform(0, 0.1)

        v_esc = ((2*G*M)/(r**2 + a**2)**(1/2))**(1/2)
        v = 1 * v_esc

        #vx, vy, vz, *_ = scalar_to_sph(v)
        vx = v
        vy =0
        vz =0
        line = [str(j) for j in [x+x0, y+y0, z+z0, vx+vx0, vy+vy0, vz+vz0, mass]]
        points.append(line)

    return points

"""

    for i in range(size):
        r = rn.uniform(0.0, box_size)

        xmax = r - abs(r*cos(phi)*cos(thet))
        xi = rn.uniform(-xmax, xmax)

        ytrans = r - (r*cos(phi)*sin(thet))
        yred = np.sqrt(r**2 - xi**2)
        ymax = ytrans if ytrans < yred else yred
        yi = 0.0 if ymax == 0 else rn.uniform(-ymax, ymax)
        
        ztrans = r - (r*sin(phi))
        zred =  np.sqrt(r**2 - xi**2 - yi**2)
        zmax = ztrans if ztrans < zred else zred
        zi = 0.0 if zmax == 0 or sin(phi) == 1. else rn.uniform(-zmax, zmax)

        t = arcsin(yi/np.sqrt(yi**2+xi**2))
        p = arcsin(zi/np.sqrt(yi**2+xi**2+zi**2))
        if str(p) == "nan" or str(t) == "nan":
            raise Exception(p, t, xi, yi, zi, xi/(yi**2+xi**2))
        v = 2.0*(((G*main_mass)/r))**(0.5)
        vxi = vx - v*sin(t)*cos(p)
        vyi = vy + v*cos(t)*cos(p)
        vzi = vz + v*sin(p)
        print(xi, yi, zi, vxi, vyi, vzi, t, p)
        line = [str(j) for j in [x+xi,y+yi,z+zi,vxi,vyi,vzi, m]]
        points.append(line)
    
    points[first_index] = [str(j) for j in [x, y, z, vx, vy, vz, main_mass]]
    return points
"""
