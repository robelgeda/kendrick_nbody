import random as rn 
from sys import argv 
from numpy import *
import numpy as np
from os import system as s

try:
    import dist
except Exception as e:
    print("Distro import failed: ", str(e))
    raise e

G = dist.G

def ot_make_points(size, steps, box_size, time_step, cut, G):
    print(size, steps, box_size, time_step, cut, G)
    print("Make init")
    points = []

    r = 50
    thet = 0 # rn.uniform(0.0000001, 2.0 * pi)  # + phi0
    phi0 = 0
    psi0 = 0

    main_mass = 1e10
    disk_mass = main_mass/10000

    v = 1 * np.sqrt(((G * (main_mass)) / r))  # / np.sqrt(2)

    points.append([str(j) for j in [0, 0, 0, 0, 0, 0, main_mass]])
    points.append([str(j) for j in [r, 0, 0, 0, v, 0, disk_mass]])


    return points


def ot_2make_points(size, steps, box_size, time_step, cut, G):
    """Edit this to init points"""
    print("Make init")
    points = []

    disk_mass = 1e10
    main_mass = 1e12
    total_main_mass = disk_mass + main_mass

    #points = [[str(j) for j in [0, 0, 0, 0, 0, 0, total_main_mass]]]

    size_mw = size//4
    size_dwarf = size - size_mw

    points = dist.disk(
        size_mw, box_size,
        0, 0, 0, disk_mass, main_mass,
        vx0=0., vy0=0., vz0=0.,
        phi0=0.0, psi0=0.0)

    z0 = box_size*4
    vy0 =  0.8 * np.sqrt(((G * (total_main_mass)) / z0))

    sphere_box_size = box_size * (14/100)
    sphere_mass = total_main_mass / 1000
    sphere_m = sphere_mass / (size_dwarf)

    points += dist.stable_sphere(
        size_dwarf, sphere_box_size,
        0., 0., z0, sphere_m, sphere_mass,
        vx=0., vy=vy0, vz=0.,)

    return points

def make_points_ot(size, steps, box_size, time_step, cut, G):
    print(size, steps, box_size, time_step, cut, G)
    print("Make init")
    points = []

    main_mass = 1.5e12
    disk_mass = main_mass/10000

    each_size = size//4
    points = dist.disk(
        each_size, box_size,
        0, 0, 0, disk_mass, main_mass,
        vx0=0., vy0=0., vz0=0.,
        phi0=0.0, psi0=0.0)

    x0 = box_size*8
    vz0 =  0.8 * np.sqrt(((G * (main_mass)) / x0))

    points += dist.disk(
        each_size,  box_size,
        x0, 0, 0, disk_mass, main_mass,
        vx0=0., vy0=vz0, vz0=0,
        phi0=0, psi0=np.pi/(2*5))

    points += dist.stable_sphere(each_size, box_size/10,
                  0, x0/4, 0, disk_mass/(each_size)/1000, main_mass/1000,
                  vx=vz0*2, vy=0., vz=0,)

    points += dist.stable_sphere(size // 4, box_size,
                            -x0, +x0*4, 0, disk_mass/(each_size)/2, main_mass/2,
                            vx=0., vy=0., vz=0.,)

    return points


def make_points_ot(size, steps, box_size, time_step, cut, G):
    print(size, steps, box_size, time_step, cut, G)
    print("Make init")
    points = []

    disk_mass = 1e12
    main_mass = disk_mass / 100000


    N = 20
    each_size = size//N
    main_mass /= N
    disk_mass /= N
    total_main_mass = disk_mass + main_mass

    points = []

    for j in range(N//2):

        r = rn.uniform(0.0, box_size)
        phi = rn.uniform(0.0000001, 2.0 * pi)
        costheta = rn.uniform(-1, 1)
        thet = arccos(costheta)

        xi = r * sin(thet) * cos(phi)
        yi = r * costheta
        zi = r * sin(thet) * sin(phi)

        vy0 = 0.8 * np.sqrt(((G * (total_main_mass)) / r))

        points += dist.stable_sphere(each_size, 1e4,
                      xi, yi, zi, disk_mass/each_size, main_mass,
                      vx=rn.uniform(0, vy0), vy=rn.uniform(0, vy0), vz=rn.uniform(0, vy0),)

    # for j in range(N//2):
    #
    #     r = rn.uniform(0.0, box_size)
    #     phi = 0#rn.uniform(0.0000001, 2.0 * pi)
    #     costheta = rn.uniform(-1, 1)
    #     thet = arccos(costheta)
    #
    #     xi = r * sin(thet) * cos(phi)
    #     yi = r * costheta
    #     zi = r * sin(thet) * sin(phi)
    #
    #     vy0 = 0.8 * np.sqrt(((G * (total_main_mass)) / r))
    #
    #     points += dist.stable_sphere(each_size, 1e4,
    #                   xi, yi, zi, disk_mass/each_size, main_mass,
    #                   vx=rn.uniform(0, vy0), vy=rn.uniform(0, vy0), vz=rn.uniform(0, vy0),)

    N2 = size - len(points)
    for j in range(N2):

        r = rn.uniform(0.0, box_size*2)
        phi = rn.uniform(0.0000001, 2.0 * pi)
        costheta = rn.uniform(-1, 1)
        thet = arccos(costheta)

        xi = r * sin(thet) * cos(phi)
        yi = r * costheta
        zi = r * sin(thet) * sin(phi)

        vy0 = 0.8 * np.sqrt(((G * (total_main_mass)) / r))
        m = total_main_mass/ N2
        points += [[str(j) for j in [xi,yi,zi,0,0,0, m]]]

    return points

def make_points(size, steps, box_size, time_step, cut, G):
    print(size, steps, box_size, time_step, cut, G)
    print("Make init")
    points = []

    main_mass = 1.5e12
    disk_mass = 1e6

    each_size = size #// 2
    points = dist.uniform_disk(
        each_size, box_size,
        0, 0, 0, disk_mass, main_mass,
        vx0=0., vy0=0., vz0=0.,
        phi0=0.0, psi0=0.0)

    #r = box_size*4
    #vy0 = 0.6 * np.sqrt(((G * (main_mass)) / r))
    #points += dist.uniform_disk(
    #each_size, box_size,
    #    r, 0, 0, disk_mass, main_mass,
    #    vx0=0, vy0=vy0, vz0=0.,
    #    phi0=0.0, psi0=0.0)

    # idx = len(points)
    # each_size = 2*size // 10
    # points += dist.stable_sphere(each_size, box_size,
    #                              0, 0, 0, disk_mass / (each_size), main_mass,
    #                              vx=0, vy=0., vz=0, )
    #
    # points[idx] = points[idx+1]

    return points
# def make_points(size, steps, box_size, time_step, cut, G):
#     points = []
#
#     R = box_size*1.5
#     num_p = int(size*0.10)
#     total_mass = 10000
#     single_mass = total_mass / num_p
#     points = collapse_shpere(points, num_p, R,
#                             0., 0., 0., single_mass)
#
#     R = box_size*4
#     v = 0.2*( (G*total_mass) / (R+1) )**(0.5)
#
#     num_p = size-len(points)
#     total_mass = 1
#     single_mass = total_mass / num_p
#
#     points = collapse_shpere(points, num_p, box_size,
#                              R, 0., 0., single_mass, vy=v)
#
#     # points = disk(points, num_p, box_size,
#     #                     R, 0., 0.,
#     #                     single_mass, 100, vy=v)
#     return points
