import random as rn 
from sys import argv 
from numpy import *
import numpy as np
from os import system as s

try:
	from point_init import make_points
except Exception as e:
	print("make_points import failed: ", str(e))
	raise e

G = 1.5607939e-22

def compile(argv):
	"""Compile Kendrick"""
	# command = "nvcc kendrick.cu -o x -lm -arch=compute_30 -O2"
	command = "nvcc kendrick.cu -o kexe -lm  -O2"

	for i in argv[2:]:
		command = command + " " + i
	print(command)
	s(command)


def simulate(points, size, steps, box_size, time_step, cut):
	"""Simulate Points"""

	assert len(points) == size, "{} != {}".format(len(points), size)

	with open(init_fn,"w") as f:
		f.write(str(size)+" "+str(steps)+" "+str(box_size)+" "+str(time_step)+" "+str(cut)+"\n")
		for line in points:
			line = " ".join(line)
			f.write(line+"\n")

	###########
	# Run Sim #
	###########
	print("./kexe %s" %init_fn)
	s("./kexe %s" %init_fn)
	print("python toVTK2.py %s" %output_fn)
	s("python toVTK2.py %s" %output_fn)


if __name__ == "__main__":
	if "-c" in argv:
		compile(argv)
		exit()
	if "-h" in argv:
		print(">> size >> steps >> box_size >> time_step >> cut >> <name>")
		exit()

	size = int(argv[1])
	steps = int(argv[2])
	box_size = int(argv[3])
	time_step = float(argv[4])
	cut = int(argv[5])

	if len(argv) == 7:
		run_name = argv[6]
	else:
		run_name = "param"

	init_fn = run_name+"_init.txt"
	output_fn = run_name + "_output"

	############################################################################

	points = make_points(size, steps, box_size, time_step, cut, G)
	simulate(points, size, steps, box_size, time_step, cut)

