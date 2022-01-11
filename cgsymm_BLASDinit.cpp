/**************************************************************************
cgsymm_BLASDinit
initialize cgsymm_BLASD
TWS, April 2021
Cuda 10.1 Version
**************************************************************************/
//#include <shrUtils.h>
//#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include "nrutil.h"

void cgsymm_BLASDinit(int nnod, int nodsegm, int matrixdim)
{
	extern int useGPU;
	extern int* nodtyp, * knowntyp, * nodelambda, ** nodnod;
	extern int* d_nodtyp, * d_knowntyp, * d_nodelambda, * d_nodnod;
	extern int* h_nodtyp, * h_knowntyp, * h_nodelambda, * h_nodnod;
	extern double* precond, ** hmat, ** kmat;
	extern double* d_precond, * d_hmat, * d_kmat;
	extern double* h_precond, * h_hmat, * h_kmat;
	extern double* h_x, * h_b, * d_x, * d_b, * d_r, * d_v, * d_p;

	int inod, i;
	const int nmem1 = matrixdim * sizeof(double);	//needed for malloc
	const int nmem2 = nnod * sizeof(double);
	const int nmem3 = nnod * nodsegm * sizeof(double);
	const int nmem4 = nnod * sizeof(int);
	const int nmem5 = nnod * nodsegm * sizeof(int);

	h_nodtyp = ivector(0, nnod - 1);
	h_knowntyp = ivector(0, nnod - 1);
	h_nodelambda = ivector(0, nnod - 1);
	h_nodnod = ivector(0, nnod * nodsegm - 1);
	h_precond = dvector(0, matrixdim - 1);
	h_hmat = dvector(0, nnod * nodsegm - 1);
	h_kmat = dvector(0, nnod * nodsegm - 1);
	h_x = dvector(0, matrixdim - 1);
	h_b = dvector(0, matrixdim - 1);

	for (inod = 1; inod <= nnod; inod++) {
		h_nodtyp[inod - 1] = nodtyp[inod];
		h_nodelambda[inod - 1] = nodelambda[inod] - 1;	//subtract 1 for 0-based indexing;
		h_knowntyp[inod - 1] = knowntyp[inod];
		for (i = 1; i <= nodsegm; i++) h_nodnod[(i - 1) * nnod + inod - 1] = nodnod[i][inod] - 1;	//subtract 1 for 0-based indexing
	}

	cudaSetDevice( useGPU-1 );	//device 
	cublasInit();
	cudaMalloc((void**)& d_r, nmem1);
	cudaMalloc((void**)& d_v, nmem1);
	cudaMalloc((void**)& d_p, nmem1);
	cudaMalloc((void**)& d_x, nmem1);
	cudaMalloc((void**)& d_b, nmem1);
	cudaMalloc((void**)& d_nodtyp, nmem4);
	cudaMalloc((void**)& d_knowntyp, nmem4);
	cudaMalloc((void**)& d_nodelambda, nmem4);
	cudaMalloc((void**)& d_nodnod, nmem5);
	cudaMalloc((void**)& d_precond, nmem1);
	cudaMalloc((void**)& d_hmat, nmem3);
	cudaMalloc((void**)& d_kmat, nmem3);

	cudaMemcpy(d_nodtyp, h_nodtyp, nmem4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_knowntyp, h_knowntyp, nmem4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nodelambda, h_nodelambda, nmem4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nodnod, h_nodnod, nmem5, cudaMemcpyHostToDevice);
}
