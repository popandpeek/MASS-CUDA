/*! \file cudaUtil.h
 * Rob Jordan

 * Utility functions for CUDA applications
 *
 * Error checking functions were adapted from
 * http://choorucode.wordpress.com/2011/03/02/cuda-error-checking/
 * and "Cuda by Example" book.h sample code.
 *
 * define CERR to turn on error checking
 */
#ifndef CUDAUTIL_H_
#define CUDAUTIL_H_

namespace mass {

/*! If a cuda error occurs, terminates the program with a descriptive error message.
 */
#define CATCH(err) __cudaCatch( err, __FILE__, __LINE__ )

/*! Tests whether a cuda error has occured and terminates the program with a descriptive error
 * message if so.
 */
#define CHECK() __cudaCheckError( __FILE__, __LINE__ )

/*! Terminates the program with a descriptive error message if a cuda error occurs.
 */
void __cudaCatch( cudaError err, const char *file, const int line );

/*! Tests whether a cuda error has occured and terminates the program with a descriptive error
 * message if so.
 */
void __cudaCheckError( const char *file, const int line );

/*! More careful checking. However, this will affect performance.*/
void __cudaCheckSync( const char *file, const int line );

/*! Provides a synchronization barrier for all devices identified by the given device ids.
 */
void syncDevices( int *devices, int ngpu );

/*! Get all CUDA devices on the system regardless of their characteristics.
 * devices: a pointer to an array of device indices that will be filled by this method.
 * returns: the count of cuda-enabled GPUs on this system.
 */
int getAllDevices( int **devices );

/*! A combination of calloc() and cudaMalloc(). Allocates device memory and sets the memory to zero.
 * Performed asynchronously.
 */
cudaError_t cudaCallocAsync( void **devPtr, size_t size, cudaStream_t stream );
} /* namespace mass */
#endif //CUDAUTIL_H_
