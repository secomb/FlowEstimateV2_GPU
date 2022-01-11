/**************************************************************************
cgsymm_BLASDend - double precision
end cgsymm_BLASD
TWS, April 2021
Cuda 10.1 Version
**************************************************************************/
//#include <shrUtils.h>
//#include <cutil_inline.h>
#include <cuda_runtime.h>
#include "nrutil.h"

void cgsymm_BLASDend(int nnod, int nodsegm, int matrixdim)
{	
	extern int* d_nodtyp, * d_knowntyp, * d_nodelambda, * d_nodnod;
	extern int* h_nodtyp, * h_knowntyp, * h_nodelambda, * h_nodnod;
	extern double* d_precond, * d_hmat, * d_kmat;
	extern double* h_precond, * h_hmat, * h_kmat;
	extern double* h_x, * h_b, * d_x, * d_b, * d_r, * d_v, * d_p;

	cudaFree(d_nodtyp);
	cudaFree(d_knowntyp);
	cudaFree(d_nodelambda);
	cudaFree(d_nodnod);
	cudaFree(d_precond);
	cudaFree(d_hmat);
	cudaFree(d_kmat);
	cudaFree(d_b);
	cudaFree(d_x);
	cudaFree(d_p);
	cudaFree(d_v);
	cudaFree(d_r);

	free_ivector(h_nodtyp, 0, nnod - 1);
	free_ivector(h_knowntyp, 0, nnod - 1);
	free_ivector(h_nodelambda, 0, nnod - 1);
	free_ivector(h_nodnod, 0, nnod * nodsegm - 1);
	free_dvector(h_precond, 0, matrixdim - 1);
	free_dvector(h_hmat, 0, nnod * nodsegm - 1);
	free_dvector(h_kmat, 0, nnod * nodsegm - 1);
	free_dvector(h_b, 0, matrixdim - 1);
	free_dvector(h_x, 0, matrixdim - 1);
}
