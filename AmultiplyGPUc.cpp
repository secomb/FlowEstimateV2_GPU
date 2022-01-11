/**************************************************************************
AmultipyGPUc.cpp
program to call AmultiplyGPU.cu on GPU
TWS, April 2021
**************************************************************************/
//#include <shrUtils.h>
//#include <cutil_inline.h>
#include "cuda_runtime.h"

extern "C" void AmultiplyGPU(double* d_input, double* d_output, double* d_precond, double* d_hmat,
	double* d_kmat, int* d_nodtyp, int* d_knowntyp, int* d_nodelambda, int* d_nodnod, int nnod);

void AmultiplyGPUc(double* d_input, double* d_output) {
	extern int nnod;
	//extern int nnod, matrixdim;
	extern int* d_nodtyp, * d_knowntyp, * d_nodelambda, * d_nodnod;
	extern double* d_precond, * d_hmat, * d_kmat;
	//extern double* h_input, * h_output, * d_input, * d_output;

	//int inod, i;
	//const int nmem1 = matrixdim * sizeof(double);
	//cudaError_t error;

	//for (i = 0; i < matrixdim; i++) h_input[i] = input[i + 1];

	//error = cudaMemcpy(d_input, h_input, nmem1, cudaMemcpyHostToDevice);
	AmultiplyGPU(d_input, d_output, d_precond, d_hmat,
		d_kmat, d_nodtyp, d_knowntyp, d_nodelambda, d_nodnod, nnod);
	//error = cudaMemcpy(h_output, d_output, nmem1, cudaMemcpyDeviceToHost);

	//for (i = 0; i < matrixdim; i++) output[i + 1] = h_output[i];
}
