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

def make_points(size, steps, box_size, time_step, cut, G):
    """Edit this to init points"""
    print("Make init")
    points = []

    disk_mass = 10
    main_mass = 4
    total_main_mass = disk_mass + main_mass

    #points = [[str(j) for j in [0, 0, 0, 0, 0, 0, total_main_mass]]]

    size_mw = size//4
    size_dwarf = size - size_mw

    points = dist.disk(
        size_mw, box_size,
        0, 0, 0, disk_mass, main_mass,
        vx0=0., vy0=0., vz0=0.,
        phi0=0.0, psi0=0.0)

    z0 = box_size*10
    vy0 =  0.8 * np.sqrt(((G * (total_main_mass)) / z0))

    sphere_box_size = box_size * (14/100)
    sphere_mass = total_main_mass / 1000
    sphere_m = sphere_mass / (size_dwarf)

    points += dist.stable_sphere(
        size_dwarf, sphere_box_size,
        0., 0., z0, sphere_m, sphere_mass,
        vx=0., vy=vy0, vz=0.,)

    return points
"""
def make_points(size, steps, box_size, time_step, cut, G):
    print(size, steps, box_size, time_step, cut, G)
    print("Make init")
    points = []

    disk_mass = 10
    main_mass = 200

    each_size = size//4
    points = new_disk(
        each_size, box_size,
        0, 0, 0, disk_mass, main_mass,
        vx0=0., vy0=0., vz0=0.,
        phi0=0.0, psi0=0.0)

    x0 = box_size*3
    vz0 =  0.7 * np.sqrt(((G * (main_mass)) / x0))

    points += new_disk(
        each_size,  box_size,
        x0, 0, 0, disk_mass, main_mass,
        vx0=0., vy0=0., vz0=vz0,
        phi0=0, psi0=np.pi/4)

    points += stable_sphere(each_size, box_size,
                  0, x0*2, 0, disk_mass/(each_size), main_mass,
                  vx=0., vy=0., vz=-vz0,
                  phi=0.0, psi=0.0)

    points += stable_sphere(size // 4, box_size,
                            x0, -x0*8, 0, disk_mass/(each_size)*5, main_mass,
                            vx=0., vy=vz0, vz=0.,
                            phi=0.0, psi=0.0)

    return points
"""

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
