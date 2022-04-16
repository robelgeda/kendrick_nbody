import random as rn 
from sys import argv 
import numpy as np
from os import system as s
from time import time
import datetime

try:
	from point_init import make_points, G
except Exception as e:
	print("make_points import failed: ", str(e))
	raise e

def compile(argv):
	"""Compile Kendrick"""
	# command = "nvcc kendrick.cu -o x -lm -arch=compute_30 -O2"
	command = "nvcc kendrick.cu -o kexe -lm  -O2"
	command = "nvcc -lm kendrick.cu -o kexe"

	for i in argv[2:]:
		command = command + " " + i
	print(command)
	s(command)


def simulate(points, size, steps, box_size, time_step, cut, run_name):
	"""Simulate Points"""

	assert len(points) == size, "{} != {}".format(len(points), size)

	init_fn = run_name + "_init.txt"
	output_fn = run_name + "_output"

	with open(init_fn,"w") as f:
		f.write(str(size)+" "+str(steps)+" "+str(box_size)+" "+str(time_step)+" "+str(cut)+"\n")
		for line in points:
			line = " ".join(line)
			f.write(line+"\n")

	###########
	# Run Sim #
	###########
	t_sim_start = time()
	print("./kexe %s" %init_fn)
	s("./kexe %s" %init_fn)
	t_sim_end = time()
	print("python toVTK2.py %s" %output_fn)
	s("python toVTK2.py %s" %output_fn)

	sim_run_time = t_sim_end-t_sim_start
	time_str = str(datetime.timedelta(seconds=sim_run_time))
	print("Run Time:", time_str)
	print("Run Time Per Step:", sim_run_time/steps)


if __name__ == "__main__":
	if "-c" in argv:
		compile(argv)
		exit()
	if "-h" in argv:
		print(">> size >> steps >> box_size >> time_step >> cut >> <name>")
		exit()

	size = int(argv[1])
	steps = int(float(argv[2]))
	box_size = float(argv[3])
	time_step = float(argv[4])
	cut = int(argv[5])

	if len(argv) == 7:
		run_name = argv[6]
	else:
		run_name = "param"

	############################################################################

	points = make_points(size, steps, box_size, time_step, cut, G)
	simulate(points, size, steps, box_size, time_step, cut, run_name)

