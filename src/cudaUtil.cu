/* cudaUtil.cu
 * Rob Jordan
 */

#include "cudaUtil.h"
#include "stdio.h"     // fprintf()
namespace mass {

void __cudaCatch(cudaError err, const char *file, const int line) {
#ifdef CERR
	if (cudaSuccess != err) {
		fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file,
				line);
		exit(EXIT_FAILURE);
	}
#endif
}

void __cudaCheckError(const char *file, const int line) {
#ifdef CERR
	__cudaCatch(cudaGetLastError(), file, line);
#endif
}

void __cudaCheckSync(const char *file, const int line) {
#ifdef CERR
	__cudaCatch(cudaGetLastError(), file, line);
	__cudaCatch(cudaDeviceSynchronize(), file, line);
#endif
}

void syncDevices(int *devices, int ngpu) {
	for (int i = 0; i < ngpu; i++) {
		CATCH(cudaSetDevice(devices[i]));
		CATCH(cudaDeviceSynchronize());
	}
}

int getAllDevices(int **devices) {
	int ngpu, *devs;
	CATCH(cudaGetDeviceCount(&ngpu));
	devs = (int *) malloc(ngpu * sizeof(*devs));
	for (int i = 0; i < ngpu; i++) {
		devs[i] = i;
	}
	*devices = devs;
	return ngpu;
}

cudaError_t cudaCallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
	cudaError_t err = cudaMalloc(devPtr, size);
	if (err == cudaSuccess) {
		err = cudaMemsetAsync(*devPtr, 0, size, stream);
	}
	return err;
}

} /* namespace mass */
