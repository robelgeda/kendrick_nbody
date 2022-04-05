// nvcc obla.cu -arch=compute_35 -lm
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <iostream>
#include <fstream>


// ******** globals ********
int steps, size;
int force_size;
float time_step, box_size;
float G = 0.000864432; //km3 kg-1 day-2 
char filename[100] = "new.dat";
//const float PI = 3.14159265359;

////////////////////////////////////////////////////////////////////////////
__global__ void calcForce(float4 *pos, float3 *vel, int size, float G, float time_step)
{
	int i = threadIdx.x + (blockIdx.x *blockDim.x);
	if(i >= size)
		return;

	float3 *vi = &vel[i];
	float4 *pi = &pos[i];

	if(pi->w < 0)
		return;

	float x, y, z;
	float dx, dy, dz, r2, a;
	float ax, ay, az;
	int j;

	x = pi->x;
	y = pi->y;
	z = pi->z;

    ax = 0.0;
    ay = 0.0;
    az = 0.0;

    float4 pj;
	for(j = 0; j < size; j++)
	{	
		pj = pos[j];
		if(j == i || pj.w < 0)
		{
			continue;	
		}

		dx = pj.x - x;
		dy = pj.y - y;
		dz = pj.z - z;

		r2 = (dx*dx)+(dy*dy)+(dz*dz); 
		//if(r2 > 10000) continue;
		a = ((G*pj.w)/(r2+4))/(sqrtf(r2)+2); 
	            
        ax += a*dx;
        ay += a*dy;
        az += a*dz;
	}

	pi->x = pi->x + vi->x*time_step + (0.5*ax*(time_step*time_step));
	vi->x = vi->x + ax*time_step;

	pi->y = pi->y + vi->y*time_step + (0.5*ay*(time_step*time_step));
	vi->y = vi->y + ay*time_step;

	pi->z = pi->z + vi->z*time_step + (0.5*az*(time_step*time_step));
	vi->z = vi->z + az*time_step;

	if((pi->x*pi->x)+(pi->y*pi->y)+(pi->z*pi->z) > 1000000000000)
	{
		pi->w = -1;
		//pi->ax = 0.0; pi->ay = 0.0; pi->az = 0.0;
		return;
	}
	
	return;
}
////////////////////////////////////////////////////////////////////////////
int toFile(float4 *pos)
{
	int i;
	float4 p;

	FILE *fx = fopen("x.dat","a");
	FILE *fy = fopen("y.dat","a");
	FILE *fz = fopen("z.dat","a");
	FILE *fm = fopen("m.dat","a");

	for(i = 0; i < size; i++)
	{
		p = pos[i];
		if(p.w > 0)
		{
			fprintf(fx, "%f\t", p.x);
			fprintf(fy, "%f\t", p.y);
			fprintf(fz, "%f\t", p.z);
			fprintf(fm, "%f\t", p.w);
		}
	}
	fprintf(fx, "\n");
	fprintf(fy, "\n");
	fprintf(fz, "\n");
	fprintf(fm, "\n");
	fclose(fx);fclose(fy);fclose(fz);fclose(fm);
	return 0;
}

void clearFile()
{
	FILE *fx = fopen("x.dat","w");
	FILE *fy = fopen("y.dat","w");
	FILE *fz = fopen("z.dat","w");
	FILE *fvx = fopen("vx.dat","w");
	FILE *fvy = fopen("vy.dat","w");
	FILE *fvz = fopen("vz.dat","w");
	FILE *fm = fopen("m.dat","w");

	fclose(fx);fclose(fy);fclose(fz);fclose(fvx);fclose(fvy);fclose(fvz);fclose(fm);
}
////////////////////////////////////////////////////////////////////////////
int main(int argc, char  ** argv)
{

	int cut; // write to file evey cut iteration 
	int t; //time 
	int i; // count

	///////////////////////////////////////////////////
	std::ifstream inFile(argv[1]);

	inFile >> size >> steps >> box_size >> time_step >> cut;
	G = G * time_step * time_step;
	printf("Size: %d\n", size);
	printf("Steps: %d\n", steps);
	printf("Box_size: %f\n", box_size);
	printf("Time_step: %f\n", time_step);
	printf("Cut: %d\n", cut);
	printf("G: %f\n", G);

	///////////////////////////////////////////////////
	printf("Init\n");
	clearFile();
	cudaSetDevice(0);
	cudaDeviceReset();

	float4 *pos, *Hpos;
	float3 *vel, *Hvel;

	Hpos = (float4*)malloc(sizeof(float4)*size);
	Hvel = (float3*)malloc(sizeof(float3)*size);

	cudaError_t rc = cudaMalloc((void**)&pos, size*sizeof(float4));
	if (rc != cudaSuccess)
	{
    	printf("Could not allocate memory 1: %d", rc);
    	return 1;
	}

	rc = cudaMalloc((void**)&vel, size*sizeof(float3));
	if (rc != cudaSuccess)
	{
    	printf("Could not allocate memory 2: %d", rc);
    	return 1;
	}

	///////////////////////////////////////////////////
	printf("Load points\n");	

	float x, y, z, vx, vy, vz, m;
	i = 0;
	while (inFile >> x >> y >> z >> vx >> vy >> vz >> m) {
		Hpos[i].x = x; Hpos[i].y = y; Hpos[i].z = z; Hpos[i].w = m;
		Hvel[i].x = vx; Hvel[i].y = vy; Hvel[i].z = vz;
		i++;		
	}	
	printf("%d\n",i);

	toFile(Hpos);
	printf("Transfer points\n");	
	cudaMemcpy(pos, Hpos, size*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(vel, Hvel, size*sizeof(float3), cudaMemcpyHostToDevice);

	
	///////////////////////////////////////////////////
	printf("GPU\n");
	int threads = 256;
	int blocks = 1+(int)size/(int)threads;
	dim3 grid(blocks,1);
	dim3 block(threads,1,1);

	toFile(Hpos);
	for(t = 0; t <= steps; t++)
	{
		printf("%d",t);

		calcForce<<<grid, block>>>(pos, vel, size, G, time_step);
		cudaThreadSynchronize();
		
		if ((t%cut) == 0 && t != 0)
		{
			printf(" %d", (int)t/cut);
			cudaMemcpy(Hpos, pos, size * sizeof(float4), cudaMemcpyDeviceToHost);
			toFile(Hpos);
		}

		printf("\n");
	}

	return 0;
}


////////////////////////////////////////////////////////////////////////////

/*
///////////////////////////////////////
Tofile:
	FILE *fvx = fopen("vx.dat","a");
	FILE *fvy = fopen("vy.dat","a");
	FILE *fvz = fopen("vz.dat","a");


			fprintf(fvx, "%f\t", pvx);
			fprintf(fvy, "%f\t", p->vy);
			fprintf(fvz, "%f\t", p->vz);


	fprintf(fvx, "\n");
	fprintf(fvy, "\n");
	fprintf(fvz, "\n");
	
	fclose(fvx);fclose(fvy);fclose(fvz);
/////////////////////////////////////////
*/