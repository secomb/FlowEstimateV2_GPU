/***********************************************************
Amultiply_GPU.cu
GPU kernel for sparse multiplication, for FlowEstimateV2
TWS April 2021
Cuda 10.1 Version
************************************************************/

__global__ void AmultiplyGPUKernel(double *d_input, double *d_output, double *d_precond, double *d_hmat,
	double *d_kmat, int *d_nodtyp, int *d_knowntyp, int *d_nodelambda, int *d_nodnod, int nnod)
{
    int inod = blockDim.x * blockIdx.x + threadIdx.x;
	int i, currentnode, ilam;
	double result1, result2;

	if(inod < nnod) {
		result1 = d_input[inod];
		if (d_knowntyp[inod] != 0) {			//not known pressure node
			if (d_knowntyp[inod] != 3) {		//has an associated constraint
				ilam = d_nodelambda[inod];
				result1 += d_input[ilam];
				result2 = d_input[inod];
			}
			for (i = 0; i < d_nodtyp[inod]; i++) {
				currentnode = d_nodnod[i * nnod + inod];
				if (d_knowntyp[currentnode] != 0) {
					result1 += d_input[currentnode] * d_hmat[i * nnod + inod] * d_precond[inod] * d_precond[currentnode];
					if (d_knowntyp[currentnode] != 3) result1 += d_input[d_nodelambda[currentnode]] * d_kmat[i * nnod + inod]
						* d_precond[inod] * d_precond[d_nodelambda[currentnode]];
					if (d_knowntyp[inod] != 3) result2 += d_input[currentnode] * d_kmat[i * nnod + inod]
						* d_precond[ilam] * d_precond[currentnode];
				}
			}
		}
		d_output[inod] = result1;
		if (d_knowntyp[inod] != 3) d_output[d_nodelambda[inod]] = result2;
	}
}

extern "C" void AmultiplyGPU(double* d_input, double* d_output, double* d_precond, double* d_hmat,
	double* d_kmat, int* d_nodtyp, int* d_knowntyp, int* d_nodelambda, int* d_nodnod, int nnod)
{
	int threadsPerBlock = 256;
	int blocksPerGrid = (nnod + threadsPerBlock - 1) / threadsPerBlock;

	AmultiplyGPUKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_precond, d_hmat,
		d_kmat, d_nodtyp, d_knowntyp, d_nodelambda, d_nodnod, nnod);
}
