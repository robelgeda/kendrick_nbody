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
char filename[100] = "new.dat";
//const float PI = 3.14159265359;

// G
// float G = 1.5607939e-22; // klyr3 / (solMass yr2)
float G = 1.5607939e-13; // lyr3 / (solMass yr2)


////////////////////////////////////////////////////////////////////////////
__global__ void update_position(float4 *pos, float3 *vel, int size, float G, float time_step)
{
	int i = threadIdx.x + (blockIdx.x *blockDim.x);
	if(i >= size)
		return;

	float3 *vi = &vel[i];
	float4 *pi = &pos[i];

	if(pi->w < 0)
		return;

	pi->x = pi->x + vi->x*time_step;
	pi->y = pi->y + vi->y*time_step;
	pi->z = pi->z + vi->z*time_step;

	if((pi->x*pi->x)+(pi->y*pi->y)+(pi->z*pi->z) > 1e18)
	{
		pi->w = -1;
		//pi->ax = 0.0; pi->ay = 0.0; pi->az = 0.0;
		return;
	}

	return;
}

__global__ void calcForce(float4 *pos, float3 *vel, float *pot,
                          int size, float G, float time_step)
{
	int i = threadIdx.x + (blockIdx.x *blockDim.x);
	if(i >= size)
		return;

	float3 *vi = &vel[i];
	float4 *pi = &pos[i];

	if(pi->w < 0)
		return;

	float x, y, z;
	float dx, dy, dz, a;
	float r1, r2, r3;
	float ax, ay, az;
	int j;

	x = pi->x;
	y = pi->y;
	z = pi->z;

    ax = 0.0;
    ay = 0.0;
    az = 0.0;
    pot[i] = 0;

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

		r2 = (dx*dx)+(dy*dy)+(dz*dz)+16;
		r1 = sqrtf(r2);
		r3 = r1 * r2;

		//if(r2 > 10000) continue;
		a = ((G*pj.w)/(r3));

        ax += a*dx;
        ay += a*dy;
        az += a*dz;

        pot[i] -= (G*pj.w)/(r1);
	}
	vi->x = vi->x + ax*time_step;
	vi->y = vi->y + ay*time_step;
	vi->z = vi->z + az*time_step;

	if((pi->x*pi->x)+(pi->y*pi->y)+(pi->z*pi->z) > 1e18)
	{
		pi->w = -1;
		//pi->ax = 0.0; pi->ay = 0.0; pi->az = 0.0;
		return;
	}

	return;
}

////////////////////////////////////////////////////////////////////////////
int toFile(float4 *pos, float3 *vel, float *pot)
{
	int i;
	float4 p;
	float3 pv;
	float pp;

	FILE *fx = fopen("x.dat","a");
	FILE *fy = fopen("y.dat","a");
	FILE *fz = fopen("z.dat","a");
	FILE *fm = fopen("m.dat","a");

	FILE *fvx = fopen("vx.dat","a");
	FILE *fvy = fopen("vy.dat","a");
	FILE *fvz = fopen("vz.dat","a");

	FILE *fp = fopen("pot.dat","a");

	for(i = 0; i < size; i++)
	{
		p = pos[i];
		pv = vel[i];
		pp = pot[i];

		if(p.w > 0)
		{
			fprintf(fx, "%e\t", p.x);
			fprintf(fy, "%e\t", p.y);
			fprintf(fz, "%e\t", p.z);
			fprintf(fm, "%e\t", p.w);

			fprintf(fvx, "%e\t", pv.x);
			fprintf(fvy, "%e\t", pv.y);
			fprintf(fvz, "%e\t", pv.z);

			fprintf(fp, "%e\t", pp);
		}
	}
	fprintf(fx, "\n");
	fprintf(fy, "\n");
	fprintf(fz, "\n");
	fprintf(fm, "\n");

	fprintf(fvx, "\n");
	fprintf(fvy, "\n");
	fprintf(fvz, "\n");

	fprintf(fp, "\n");

	fclose(fx);fclose(fy);fclose(fz);fclose(fm);
	fclose(fvx);fclose(fvy);fclose(fvz);
	fclose(fp);
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
	FILE *fp = fopen("pot.dat","w");

	fclose(fx);fclose(fy);fclose(fz);fclose(fm);
	fclose(fvx);fclose(fvy);fclose(fvz);
	fclose(fp);
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
	// G = G * time_step * time_step;
	printf("Size: %d\n", size);
	printf("Steps: %d\n", steps);
	printf("Box_size: %f\n", box_size);
	printf("Time_step: %f\n", time_step);
	printf("Cut: %d\n", cut);
	printf("G: %f\n", G);

	///////////////////////////////////////////////////
	printf("Init\n");
	clearFile();
	//cudaSetDevice(0);
	cudaDeviceReset();

	float4 *pos, *Hpos;
	float3 *vel, *Hvel;
	float *pot, *Hpot;

	Hpos = (float4*)malloc(sizeof(float4)*size);
	Hvel = (float3*)malloc(sizeof(float3)*size);
	Hpot = (float*)malloc(sizeof(float)*size);

	cudaError_t rc = cudaMalloc((void**)&pos, size*sizeof(float4));
	if (rc != cudaSuccess)
	{
    	printf("Could not allocate memory 1: %d \n", rc);
    	printf("%s \n",cudaGetErrorString(cudaGetLastError()));
    	int v;
    	cudaRuntimeGetVersion(&v);
    	printf("Version (%d)", v);
    	cudaDriverGetVersion(&v);
    	printf("Version (%d)", v);
    	return 1;
	}

	rc = cudaMalloc((void**)&vel, size*sizeof(float3));
	if (rc != cudaSuccess)
	{
    	printf("Could not allocate memory 2: %d\n", rc);
    	return 1;
	}

	rc = cudaMalloc((void**)&pot, size*sizeof(float));
	if (rc != cudaSuccess)
	{
    	printf("Could not allocate memory 3: %d\n", rc);
    	return 1;
	}

	///////////////////////////////////////////////////
	printf("Load points\n");

	float x, y, z, vx, vy, vz, m;
	i = 0;
	while (inFile >> x >> y >> z >> vx >> vy >> vz >> m) {
		Hpos[i].x = x; Hpos[i].y = y;Hpos[i].z = z;Hpos[i].w = m;
		Hvel[i].x = vx; Hvel[i].y = vy ;Hvel[i].z = vz;
		Hpot[i] = -1;
		i++;
	}
	printf("%d\n",i);

	printf("Transfer points\n");
	cudaMemcpy(pos, Hpos, size*sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(vel, Hvel, size*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(pot, Hpot, size*sizeof(float),  cudaMemcpyHostToDevice);


	///////////////////////////////////////////////////
	printf("GPU\n");
	int threads = 1024;
	int blocks = (int)(ceil(size/threads))+1;
	dim3 grid(blocks,1);
	dim3 block(threads,1,1);

	toFile(Hpos, Hvel, Hpot);
	for(t = 0; t <= steps; t++)
	{
		//printf("%d",t);

		calcForce<<<grid, block>>>(pos, vel, pot, size, G, time_step);
		cudaDeviceSynchronize();
		update_position<<<grid, block>>>(pos, vel, size, G, time_step);
		cudaDeviceSynchronize();

		if ((t%cut) == 0 && t != 0)
		{
			printf(" %d", (int)t/cut);
			cudaMemcpy(Hpos, pos, size * sizeof(float4), cudaMemcpyDeviceToHost);
			cudaMemcpy(Hvel, vel, size * sizeof(float3), cudaMemcpyDeviceToHost);
			cudaMemcpy(Hpot, pot, size * sizeof(float),  cudaMemcpyDeviceToHost);
			toFile(Hpos, Hvel, Hpot);
			printf("\n");
		}

		//printf("\n");
	}

	return 0;
}

