/**************************************************************************
cgsymm_BLASD - double precision
Conjugate gradient solver for symmetric
system of linear equations:  aa(i,j)*x(j) = b(i)
Version using BLAS library on GPU
TWS, April 2021
Cuda 10.1 Version
**************************************************************************/
//#include <shrUtils.h>
//#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas.h>
#include "nrutil.h"

//void AmultiplyGPUc(double* input, double* output);
extern "C" void AmultiplyGPU(double* d_input, double* d_output, double* d_precond, double* d_hmat,
	double* d_kmat, int* d_nodtyp, int* d_knowntyp, int* d_nodelambda, int* d_nodnod, int nnod);

double cgsymm_BLASD(double* b, double* x, int n, double eps, int itmax)
{
	extern int nnod;
	extern int* d_nodtyp, * d_knowntyp, * d_nodelambda, * d_nodnod;
	extern double* h_b, * h_x, * d_b, * d_x, * d_r, * d_p, * d_v;
	extern double* d_precond, * d_hmat, * d_kmat;

	const int nmem1 = n * sizeof(double);
	int i, jj, j;
	double rsold, rsnew, pv, rsoldpv, rsnewold;
	cudaError_t error;

	for (j = 0; j < n; j++) {
		h_x[j] = x[j + 1];
		h_b[j] = b[j + 1];
	}
	error = cudaMemcpy(d_x, h_x, nmem1, cudaMemcpyHostToDevice);	//copy variables to GPU
	error = cudaMemcpy(d_b, h_b, nmem1, cudaMemcpyHostToDevice);

	AmultiplyGPU(d_x, d_r, d_precond, d_hmat, d_kmat, d_nodtyp, d_knowntyp, d_nodelambda, d_nodnod, nnod);	//r=Ax
	//r[i] = b[i] - r[i];
	cublasDscal(n, -1.f, d_r, 1);
	cublasDaxpy(n, 1.f, d_b, 1, d_r, 1);
	//p[i] = r[i]; //  initialize p as r
	cublasDcopy(n, d_r, 1, d_p, 1);
	//rsold = dot(r, r, n);
	rsold = cublasDdot(n, d_r, 1, d_r, 1);
	jj = 0;
	do {
		//AmultiplyGPUc(d_p, d_v);	//v=Ap
		AmultiplyGPU(d_p, d_v, d_precond, d_hmat, d_kmat, d_nodtyp, d_knowntyp, d_nodelambda, d_nodnod, nnod);	//v=Ap
		//pv = dot(p, v, n);
		pv = cublasDdot(n, d_p, 1, d_v, 1);
		//x[i] += rsold / pv * p[i];
		//r[i] -= rsold / pv * v[i];
		rsoldpv = rsold / pv;
		cublasDaxpy(n, rsoldpv, d_p, 1, d_x, 1);
		cublasDaxpy(n, -rsoldpv, d_v, 1, d_r, 1);
		//rsnew = dot(r, r, n);
		rsnew = cublasDdot(n, d_r, 1, d_r, 1);
		//p[i] = r[i] + rsnew / rsold * p[i];
		rsnewold = rsnew / rsold;
		cublasDscal(n, rsnewold, d_p, 1);
		cublasDaxpy(n, 1.f, d_r, 1, d_p, 1);
		jj++;
		rsold = rsnew;
		if (jj % 10000 == 0) printf("cgsymm: %i %e\n", jj, rsnew);
	} while (rsnew > n * eps * eps && jj < itmax);
	printf("cgsymm_BLASD: %i %e\n", jj, rsnew);
	cudaMemcpy(h_x, d_x, nmem1, cudaMemcpyDeviceToHost);	//bring back x for final result
	for (i = 0; i < n; i++) x[i + 1] = h_x[i];
	return rsnew;
}
