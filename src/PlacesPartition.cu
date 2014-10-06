/**
*  @file PlacesPartition.cpp
*  @author Nate Hart
*
*  @section LICENSE
*  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
*/

#define THREADS_PER_BLOCK 512

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "Dispatcher.h"
#include "Places.h"
#include "PlacesPartition.h"

namespace mass {


        PlacesPartition::PlacesPartition ( int handle, int rank, int numElements, int ghostWidth,
                          int n, int *dimensions ) :
                          hPtr ( NULL ), dPtr ( NULL ), handle ( handle ), rank ( rank ), numElements (
                          numElements ), isloaded ( false ) {
            Tsize = Mass::getPlaces ( handle )->getTsize ( );
            setGhostWidth ( ghostWidth, n, dimensions );
            setIdealDims ( );
        }

        /**
        *  Destructor
        */
        PlacesPartition::~PlacesPartition ( ) { }

        /**
        *  Returns the number of place elements in this partition.
        */
        int PlacesPartition::size ( ) {
            int numRanks = Mass::getPlaces ( handle )->getNumPartitions ( );
            if ( 1 == numRanks ) {
                return numElements;
            }

            int retVal = numElements;
            if ( 0 == rank || numRanks - 1 == rank ) {
                // there is only one ghost width on an edge rank
                retVal -= ghostWidth;
            } else {
                retVal -= 2 * ghostWidth;
            }

            return retVal;
        }

        /**
        *  Returns the number of place elements and ghost elements.
        */
        int PlacesPartition::sizePlusGhosts ( ) {
            return numElements;
        }

        /**
        *  Gets the rank of this partition.
        */
        int PlacesPartition::getRank ( ) {
            return rank;
        }

        /**
        *  Returns an array of the Place elements contained in this PlacesPartition object. This is an expensive
        *  operation since it requires memory transfer.
        */
        void *PlacesPartition::hostPtr ( ) {
            char *retVal = ( char* ) hPtr;
            if ( rank > 0 ) {
                retVal += ghostWidth * Tsize;
            }
            return ( void* ) retVal;
        }

        /**
        *  Returns a pointer to the first element, if this is rank 0, or the left ghost rank, if this rank > 0.
        */
        void *PlacesPartition::hostPtrPlusGhosts ( ) {
            return hPtr;
        }

        /**
        *  Returns the pointer to the GPU data. NULL if not on GPU.
        */
        void *PlacesPartition::devicePtr ( ) {
            return dPtr;
        }

        void PlacesPartition::setDevicePtr ( void *places ) {
            dPtr = places;
        }

        /**
        *  Returns the handle associated with this PlacesPartition object that was set at construction.
        */
        int PlacesPartition::getHandle ( ) {
            return handle;
        }

        /**
        *  Sets the start and number of places in this partition.
        */
        void PlacesPartition::setSection ( void *start ) {
            hPtr = start;
        }

        void PlacesPartition::setQty ( int qty ) {
            numElements = qty;
            setIdealDims ( );
        }

        bool PlacesPartition::isLoaded ( ) {
            return isloaded;
        }

        void PlacesPartition::setLoaded ( bool loaded ) {
            isloaded = loaded;
        }

        void PlacesPartition::makeLoadable ( ) {
            if ( !loadable ) {
                if ( dPtr != NULL ) {
                    cudaFree ( dPtr );
                }

                cudaMalloc ( ( void** ) &dPtr, Tsize * sizePlusGhosts ( ) );
                loadable = true;
            }
        }

        void PlacesPartition::load ( cudaStream_t stream ) {
            makeLoadable ( );

            cudaMemcpyAsync ( dPtr, hPtr, Tsize * sizePlusGhosts ( ),
                              cudaMemcpyHostToDevice, stream );
            isloaded = true;
        }

        bool PlacesPartition::retrieve ( cudaStream_t stream, bool freeOnRetrieve ) {
            bool retreived = isloaded;

            if ( isloaded ) {
                cudaMemcpyAsync ( hPtr, dPtr, Tsize * sizePlusGhosts ( ),
                                  cudaMemcpyDeviceToHost, stream );
            }

            if ( freeOnRetrieve ) {
                cudaFree ( dPtr );
                loadable = false;
                dPtr = NULL;
                isloaded = false;
            }

            return retreived;
        }

        int PlacesPartition::getGhostWidth ( ) {
            return ghostWidth;
        }

        void PlacesPartition::setGhostWidth ( int width, int n, int *dimensions ) {
            ghostWidth = width;

            // start at 1 because we never want to factor in x step
            for ( int i = 1; i < n; ++i ) {
                ghostWidth += dimensions[ i ];
            }
        }

        void PlacesPartition::updateLeftGhost ( void *ghost, cudaStream_t stream ) {
            if ( rank > 0 ) {
                if ( isloaded ) {
                    cudaMemcpyAsync ( dPtr, ghost, Tsize * ghostWidth,
                                      cudaMemcpyHostToDevice, stream );
                } else {
                    memcpy ( hPtr, ghost, Tsize * ghostWidth );
                }
            }
        }

        void PlacesPartition::updateRightGhost ( void *ghost, cudaStream_t stream ) {
            int numRanks = Mass::getPlaces ( handle )->getNumPartitions ( );
            if ( rank < numRanks - 1 ) {
                if ( isloaded ) {
                    char *ghostSrc = ( char* ) dPtr;
                    ghostSrc += numElements * Tsize;
                    cudaMemcpyAsync ( ghostSrc, ghost, Tsize * ghostWidth,
                                      cudaMemcpyHostToDevice, stream );
                } else {
                    char *ghostSrc = ( char* ) hPtr + ( ghostWidth + numElements ) * Tsize;
                    memcpy ( ghostSrc, ghost, Tsize * ghostWidth );
                }
            }
        }

        void *PlacesPartition::getLeftBuffer ( ) {
            if ( isloaded ) {
                char *bufSrc = ( char* ) dPtr + ghostWidth*Tsize;
                cudaMemcpy ( hPtr, bufSrc, Tsize * ghostWidth,
                             cudaMemcpyDeviceToHost );
            }

            return ( ( char* ) hPtr ) + ghostWidth*Tsize;
        }

        void *PlacesPartition::getRightBuffer ( ) {
            if ( isloaded ) {
                char *bufSrc = ( char* ) dPtr + numElements*Tsize;
                cudaMemcpy ( hPtr, bufSrc, Tsize * ghostWidth, cudaMemcpyDeviceToHost );
            }
            return ( ( char* ) hPtr ) + numElements*Tsize;
        }

        dim3 PlacesPartition::blockDim ( ) {
            return dims[ 0 ];
        }

        dim3 PlacesPartition::threadDim ( ) {
            return dims[ 1 ];
        }

        void PlacesPartition::setIdealDims ( ) {
            int numBlocks = ( numElements - 1 ) / THREADS_PER_BLOCK + 1;
            dim3 blockDim ( numBlocks, 1, 1 );

            int nThr = ( numElements - 1 ) / numBlocks + 1;
            dim3 threadDim ( nThr, 1, 1 );

            dims[ 0 ] = blockDim;
            dims[ 1 ] = threadDim;
        }

        int PlacesPartition::getPlaceBytes ( ) {
            return Tsize;
        }

} /* namespace mass */
