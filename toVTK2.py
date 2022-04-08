import numpy as np
import os, math
from sys import argv
def vtk (fn, xi, yi, zi, ri, vxi, vyi, vzi):
    npoints = len(ri)
    #print(npoints)
    #try:
    with open(fn+".vtk","w") as f:
        f.write('# vtk DataFile Version 3.0'+ "\n")
        f.write('vtk output'+ "\n")
        f.write('ASCII'+ "\n")
        f.write('DATASET UNSTRUCTURED_GRID'+ "\n")
        f.write('POINTS %d float' %(npoints)+ "\n")
        for i in range(npoints):
            f.write(" ".join([str(xi[i]), str(yi[i]), str(zi[i])])+ "\n")

        f.write(''+ "\n")
        f.write('CELLS %d %d' %(npoints, npoints*3 )+ "\n")
        u = 0
        for i in range(npoints):
            f.write('2'+" "+str(u)+" "+str(0)+ "\n")
            u = u + 1
        f.write(''+ "\n")
        f.write('CELL_TYPES %d' %(npoints)+ "\n" )
        for i in range(npoints): # ignore first line (final single halo)
            f.write('1'+ "\n")

        f.write(''+ "\n")
        f.write('POINT_DATA %d' %(npoints)+ "\n")
        f.write('FIELD FieldData %d' %2+ "\n")
        f.write('Mass 1 %d double' %(npoints)+ "\n")
        for r in ri:
            f.write(str(r/50)+ "\n")

        f.write('Velocity 3 %d double' % (npoints) + "\n")
        for i in range(npoints):
            f.write(" ".join([str(vxi[i]), str(vyi[i]), str(vzi[i])]) + "\n")
'''except:
        os.system("rm %s.vtk" %fn)'''

def ply(fn, xi, yi, zi, ri):
    npoints = len(ri)
    #print(npoints)
    with open(fn + ".ply", "w") as f:
        f.write('ply' + "\n")
        f.write('format ascii 1.0 \n')
        f.write('element vertex {}'.format(npoints))
        f.write('''property float x\n
        property float y\n
        property float z\n
        end_header\n''')

        for i in range(npoints):
            f.write(" ".join([str(xi[i]), str(yi[i]), str(zi[i])])+ "\n")



file_name = argv[1] #input("File: ")
x = []
y = []
z = []
vx = []
vy = []
vz = []
r = []
print("x.dat")
with open("x.dat") as f:
    for line in f:
        x.append(np.fromstring(line, dtype = float, sep = "\t"))

print("y.dat")
with open("y.dat") as f:
    for line in f:
        y.append(np.fromstring(line, dtype = float, sep = "\t"))

print("z.dat")
with open("z.dat") as f:
    for line in f:
        z.append(np.fromstring(line, dtype = float, sep = "\t"))

print("m.dat")
with open("m.dat") as f:
    for line in f:
        r.append(np.fromstring(line, dtype = float, sep = "\t"))

print("vx.dat")
with open("vx.dat") as f:
    for line in f:
        vx.append(np.fromstring(line, dtype = float, sep = "\t"))

print("vy.dat")
with open("vy.dat") as f:
    for line in f:
        vy.append(np.fromstring(line, dtype = float, sep = "\t"))

print("vz.dat")
with open("vz.dat") as f:
    for line in f:
        vz.append(np.fromstring(line, dtype = float, sep = "\t"))

step = len(r)
folder = file_name.replace(".dat","")
os.system("mkdir -p %s" %folder)
tf = open("rate.dat","w")
print("Length: ", len(r[0]))
for i in range(step):
    #print(i, "out of", step)
    xi = x[i]
    yi = y[i]
    zi = z[i]

    vxi = vx[i]
    vyi = vy[i]
    vzi = vz[i]

    ri = r[i]
    print(i, "out of", step - 1, ":", len(xi))
    tf.write(str(len(xi))+"\n")
    fn = folder+"/"+folder + str(i)
    vtk(fn, xi, yi, zi, ri, vxi, vyi, vzi)

print("Length: ", len(r[0]))
