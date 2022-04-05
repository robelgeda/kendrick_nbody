import random as rn 
from sys import argv 
from numpy import *
import numpy as np
from os import system as s


if "-c" in argv:
	# command = "nvcc simXX.cu -o x -lm -arch=compute_30 -O2"
	command = "nvcc kendrick.cu -o x -lm  -O2"

	for i in argv[2:]:
		command = command + " "+i
	print(command)
	s(command)
	exit()
if "-h" in argv:
	print(">> size >> steps >> box_size >> time_step >> cut >> <fn>")
	exit()

size = int(argv[1])
steps = int(argv[2])
box_size = int(argv[3])
time_step = float(argv[4])
cut = int(argv[5])

if len(argv) == 7:
	fn = argv[6]
else:
	fn = "param"


############################################################################
#>> x >> y >> z >> vx >> vy >> vz >> m
print("Make init")
points = []
G = 0.000864432


a = box_size/0.64

disk_mass = 200
mass = disk_mass / size
mainmass = 200

#disk
dens = size/(box_size*box_size)
r_list = []
for i in range(size):
	r = rn.uniform(0.0, box_size)+10
	phi = rn.uniform(0.0000001, 2.0 * pi)
	costheta = 0.0#rn.uniform(-1,1)
	thet = arccos(costheta)

	x = r*sin(thet)*cos(phi)
	y = r*costheta
	z = r*sin(thet)*sin(phi)

	m0 = len(np.where(np.array(r_list) < r)[0]) *  mass
	v = 1.05*np.sqrt(((G*(mainmass + m0))/r)) #/ np.sqrt(2)
	vx = -v*sin(thet)*sin(phi)
	vy = v*costheta
	vz = v*sin(thet)*cos(phi)
	line = [str(j) for j in [x,y,z,vx,vy,vz,mass]]
	points.append(line)
	r_list.append(r)

points[0] = [str(j) for j in [0,0,0,0,0,0,mainmass]]

'''

######################
2 disks 
####################
a = box_size/0.64

disk_mass = 1000
mass = disk_mass / size * 2
mainmass = 2000

#disk
dens = size/(box_size*box_size)
for i in range(size//2):
	r = rn.uniform(0.0, box_size)+0.1
	phi = rn.uniform(0.0000001, 2.0 * pi)
	costheta = 0.0#rn.uniform(-1,1)
	thet = arccos(costheta)

	x = r*sin(thet)*cos(phi)
	y = r*costheta
	z = r*sin(thet)*sin(phi)

	v = np.sqrt(((G*mainmass)/r)) #/ np.sqrt(2)
	vx = -v*sin(thet)*sin(phi)
	vy = v*costheta
	vz = v*sin(thet)*cos(phi)
	line = [str(j) for j in [x,y,z,vx,vy,vz,mass]]
	points.append(line)

r0 = box_size*5.0
v0 = 0.5*(((G*5000)/r0))**(0.5)
for i in range(size//2):
	r = rn.uniform(0.0, box_size)+0.1
	phi = rn.uniform(0.0000001, 2.0 * pi)
	costheta = 0.0#rn.uniform(-1,1)
	thet = arccos(costheta)

	x = r*sin(thet)*cos(phi)
	y = r*costheta
	z = r*sin(thet)*sin(phi)

	v = np.sqrt(((G*mainmass)/r))
	vx = -v*sin(thet)*sin(phi)
	vy = v*costheta
	vz = v*sin(thet)*cos(phi)
	line = [str(j) for j in [x+r0,y,z,vx,vy,vz+v0,mass]]
	points.append(line)

#v = ((G*(mainmass)/100))**(0.5)
points[0] = [str(j) for j in [0,0,0,0,0,0,mainmass]]
points[1] = [str(j) for j in [r0,0,0,0,0,v0,mainmass]]
#points[1] = [str(j) for j in [-100,0,0,0,0,-v,mainmass/2]]


##############################################################


for i in range(int(size//2)):
	#mass = abs(rn.gauss(0, 10000))
	r = random.uniform(0, box_size)
	phi = rn.uniform(0.0, 2.0 * pi)
	costheta = rn.uniform(-1,1)
	thet = arccos(costheta)
	 
	x = r*sin(thet)*cos(phi)
	y = r*sin(thet)*sin(phi)
	z = r*costheta

	v = random.uniform(0.0, 0.001)
	phi = rn.uniform(0.0, 2.0 * pi)
	costheta = rn.uniform(-1,1)
	thet = arccos(costheta)
	vm = 0.0
	vx = v*sin(thet)*cos(phi) + vm*cos(phi)
	vy = v*sin(thet)*sin(phi) + vm*sin(phi)
	vz = v*costheta

	line = [str(j) for j in [x,y,z,vx,vy,vz,mass]]
	points.append(line)

box_size = box_size
for i in range(size//2):
	#mass = abs(rn.gauss(0, 10000))
	x = random.uniform(-box_size, box_size)
	y = random.uniform(-box_size, box_size)
	z = random.uniform(-box_size, box_size)

	v = random.uniform(0.0, 0.001)
	phi = rn.uniform(0.0, 2.0 * pi)
	costheta = rn.uniform(-1,1)
	thet = arccos(costheta)
	vm = 0.0
	vx = v*sin(thet)*cos(phi) + vm*cos(phi)
	vy = v*sin(thet)*sin(phi) + vm*sin(phi)
	vz = v*costheta

	line = [str(j) for j in [x,y,z,vx,vy,vz,mass]]
	points.append(line)

#points[1] = [str(j) for j in [5000.0,0.0,0.0,0.0,0.0,0.0,10000000]]

box_size = 1000
mass = 7
for i in range(1000):
	r = random.uniform(0, box_size) 
	phi = rn.uniform(0.0, 2.0 * pi)
	costheta = rn.uniform(-1,1)
	thet = arccos(costheta)
	 
	x = r*sin(thet)*cos(phi) + 4000
	y = r*sin(thet)*sin(phi)
	z = r*costheta

	v = random.uniform(0.0, 0.01)
	phi = rn.uniform(0.0, 2.0 * pi)
	costheta = rn.uniform(-1,1)
	thet = arccos(costheta)
	vx = v*sin(thet)*cos(phi)
	vy = v*sin(thet)*sin(phi) + 0.1
	vz = v*costheta

	line = [str(j) for j in [x,y,z,vx,vy,vz,mass]]
	points.append(line)


a = box_size/0.64
mass = 7
for i in range(size):
	r = a / sqrt( random.uniform(0, 1)**(-2.0 / 3.0) - 1)
	phi = rn.uniform(0.0, 2.0 * pi)
	costheta = rn.uniform(-1,1)
	thet = arccos(costheta)
	 
	x = r*sin(thet)*cos(phi)
	y = r*sin(thet)*sin(phi)
	z = r*costheta
	vx = 0.0
	vy = 0.1
	while vy > vx*vx*(1.0-vx*vx)**3.5:
		vx = rn.uniform(0,1)
		vy = rn.uniform(0,0.1)
		
	v = vx* sqrt((2*G*mass*size)/sqrt((r**2)+(a**2)))
	phi = rn.uniform(0.0, 2.0 * pi)
	costheta = rn.uniform(-1,1)
	thet = arccos(costheta)
	vx = v*sin(thet)*cos(phi)
	vy = v*sin(thet)*sin(phi)
	vz = v*costheta
	line = [str(j) for j in [x,y,z,vx,vy,vz,mass]]
	points.append(line)


#disk
dens = size/(box_size*box_size)
for i in range(size):
	r = rn.uniform(0.0, box_size)
	phi = rn.uniform(0.0000001, 2.0 * pi)
	costheta = 0.0#rn.uniform(-1,1)
	thet = arccos(costheta)

	x = r*sin(thet)*cos(phi)
	y = r*costheta
	z = r*sin(thet)*sin(phi)
	mainmass = 1000000.0
	v = (((G*mainmass)/r))**(0.5)
	vx = -v*sin(thet)*sin(phi)
	vy = v*costheta
	vz = v*sin(thet)*cos(phi)
	line = [str(j) for j in [x,y,z,vx,vy,vz,1.0]]
	points.append(line)

#v = ((G*(mainmass)/100))**(0.5)
points[0] = [str(j) for j in [0,0,0,0,0,0,mainmass]]
#points[1] = [str(j) for j in [-100,0,0,0,0,-v,mainmass/2]]


x = box_size*0.5
y = 0.0
z = 0.0
mainmass = 1000000.0
v = ((G*(mainmass)/x))**(0.5)
vx = 0.0
vy = 0.0
vz = v
points[1] = [str(j) for j in [x,y,z,vx,vy,vz,(mainmass/500)]]

x = box_size*0.3
y = 0.0
z = 0.0
mainmass = 1000000.0
v = (((G*(mainmass+(mainmass/1047)))/x))**(0.5)
vx = 0.0
vy = 0.0
vz = v
points[1] = [str(j) for j in [x,y,z,vx,vy,vz,(mainmass/1047)]]

x = box_size*0.6
y = 0.0
z = 0.0
mainmass = 1000000.0
v = (((G*(mainmass+(mainmass*0.000284)))/x))**(0.5)
vx = 0.0
vy = 0.0
vz = v
points[2] = [str(j) for j in [x,y,z,vx,vy,vz,(mainmass*0.000284)]]

#//points[1] = init(1000.0, 2000.0, 0.0, 0.0, -5.0, 0.0, 2*M);

for k in range(int(size/5000)):
	xk = rn.uniform(-box_size, box_size)
	yk = rn.uniform(-box_size, box_size)
	zk = rn.uniform(-box_size, box_size)
	for i in range(k*5000,(k+1)*5000):
		r = rn.uniform(0.0, 2000)
		phi = rn.uniform(0.0000001, 2.0 * pi)
		costheta = 0#rn.uniform(-1,1)
		thet = arccos(costheta)

		x = r*sin(thet)*cos(phi)
		y = r*costheta
		z = r*sin(thet)*sin(phi)

		v = (((G*1000000.0)/r))**(0.5)
		vx = -v*sin(thet)*sin(phi)
		vy = v*costheta
		vz = v*sin(thet)*cos(phi)
		
		line = [str(j) for j in [x+xk,y+yk,z+zk,vx,vy,vz,10.0]]
		points.append(line)

	points[k] = [str(j) for j in [0+xk,0+yk,0+zk,0,0,0,1000000.0]]

for i in range(size):
	r = rn.uniform(0.0, box_size)
	phi = rn.uniform(0.0000001, 2.0 * pi)
	costheta = rn.uniform(-1,1)
	thet = arccos(costheta)

	x = r*sin(thet)*cos(phi)
	y = r*costheta
	z = r*sin(thet)*sin(phi)
	v = rn.uniform(-1.0, 1.0)
	vx = v*sin(thet)*cos(phi)
	vy = v*costheta
	vz = v*sin(thet)*sin(phi)
	line = [str(j) for j in [x,y,z,vx,vy,vz,10.0]]
	points.append(line)
'''

with open(fn,"w") as f:
	line = []
	f.write(str(size)+" "+str(steps)+" "+str(box_size)+" "+str(time_step)+" "+str(cut)+"\n")
	for line in points:
		line = " ".join(line)
		f.write(line+"\n")

############################################################################
print("./x %s" %fn)
s("./x %s" %fn)
print("python toVTK2.py %s" %fn)
s("python toVTK2.py %sf" %fn)

