/**
*  @file AgentsPartition.cpp
*  @author Nate Hart
*
*  @section LICENSE
*  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
*/

#include <string>
#include <vector>

#include "AgentsPartition.h"
#include "Agents.h"
#include "cudaUtil.h"
#include "Mass.h"


namespace mass {

	void *AgentsPartition::movePtr(void *ptr, int nElements){
		char* tmp = (char*) ptr;
		tmp += nElements * Tbytes;
		return tmp;
	}


	AgentsPartition::AgentsPartition ( int handle, void *argument, int argument_size, Agents *agents,
            int numElements ){
		hPtr = NULL;
		dPtr = NULL;
		this->handle = handle;
		this->rank = rank;
		this->numElements = numElements;
		this->isloaded = false;
        Tbytes = Mass::getAgents ( handle )->getTsize ( );;
        setIdealDims ( );
    }

    /**
    *  Destructor
    */
    AgentsPartition::~AgentsPartition ( ) { }

    /**
    *  Returns the number of elements in this partition.
    */
    int AgentsPartition::size ( ) {
        int numRanks = Mass::getAgents ( handle )->getNumPartitions ( );
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
    *  Returns the number of elements and ghost elements.
    */
    int AgentsPartition::sizePlusGhosts ( ) {
        return numElements;
    }

    /**
    *  Gets the rank of this partition.
    */
    int AgentsPartition::getRank ( ) {
        return rank;
    }

    /**
    *  Returns an array of the elements contained in this Partition.
    */
    void *AgentsPartition::hostPtr ( ) {
        void *retVal = hPtr;
        if ( rank > 0 ) {
            retVal = movePtr(retVal,ghostWidth);
        }
        return retVal;
    }

    /**
    *  Returns a pointer to the first element, if this is rank 0, or the left ghost rank, if this rank > 0.
    */
    void *AgentsPartition::hostPtrPlusGhosts ( ) {
        return hPtr;
    }

    /**
    *  Returns the pointer to the GPU data. NULL if not on GPU.
    */
    void *AgentsPartition::devicePtr ( ) {
        return dPtr;
    }

    void AgentsPartition::setDevicePtr ( void *agents ) {
        dPtr = agents;
    }

    /**
    *  Returns the handle associated with this AgentsPartition object that was set at construction.
    */
    int AgentsPartition::getHandle ( ) {
        return handle;
    }

    /**
    *  Sets the start and number of agents in this partition.
    */
    void AgentsPartition::setSection ( void *start ) {
        hPtr = start;
    }

    void AgentsPartition::setQty ( int qty ) {
        numElements = qty;
        setIdealDims ( );
    }

    bool AgentsPartition::isLoaded ( ) {
        return isloaded;
    }

    void AgentsPartition::setLoaded ( bool loaded ) {
        isloaded = true;
    }

    void AgentsPartition::makeLoadable ( ) {
        if ( !loadable ) {
            if ( dPtr != NULL ) {
                cudaFree ( dPtr );
            }

            cudaMalloc ( ( void** ) &dPtr, Tbytes * sizePlusGhosts ( ) );
            loadable = true;
        }
    }

    void *AgentsPartition::load ( cudaStream_t stream ) {
        makeLoadable ( );

        cudaMemcpyAsync ( dPtr, hPtr, Tbytes * sizePlusGhosts ( ),
                          cudaMemcpyHostToDevice, stream );
        isloaded = true;
        return devicePtr ( );
    }

    bool AgentsPartition::retrieve ( cudaStream_t stream, bool freeOnRetrieve ) {
        bool retreived = isloaded;

        if ( isloaded ) {
            cudaMemcpyAsync ( hPtr, dPtr, Tbytes * sizePlusGhosts ( ),
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

    int AgentsPartition::getGhostWidth ( ) {
        return ghostWidth;
    }

    void AgentsPartition::setGhostWidth ( int width, int n, int *dimensions ) {
        // agent collections are always 1D
        ghostWidth = width;
    }

    // TODO Do these ghost updates do what I want?
    // look at having them move the data to a destination pointer
    void AgentsPartition::updateLeftGhost ( void *ghost, cudaStream_t stream ) {
        if ( rank > 0 ) {
            if ( isloaded ) {
                cudaMemcpyAsync ( dPtr, ghost, Tbytes * ghostWidth,
                                  cudaMemcpyHostToDevice, stream );
            } else {
                memcpy ( hPtr, ghost, Tbytes * ghostWidth );
            }
        }
    }

    void AgentsPartition::updateRightGhost ( void *ghost, cudaStream_t stream ) {
        int numRanks = Mass::getAgents ( handle )->getNumPartitions ( );
        if ( rank < numRanks - 1 ) {
            if ( isloaded ) {
                cudaMemcpyAsync ( movePtr(dPtr, numElements) , ghost, Tbytes * ghostWidth,
                                  cudaMemcpyHostToDevice, stream );
            } else {
                memcpy ( movePtr(hPtr, ghostWidth + numElements), ghost,
                         Tbytes * ghostWidth );
            }
        }
    }

    // TODO add a pointer param so buffers and ghosts can be copied directly where they need to go
    void *AgentsPartition::getLeftBuffer ( ) {
        if ( isloaded ) {
            cudaMemcpy ( hPtr, movePtr(dPtr,ghostWidth), Tbytes * ghostWidth,
                         cudaMemcpyDeviceToHost );
        }

        return movePtr(hPtr, ghostWidth);
    }

    void *AgentsPartition::getRightBuffer ( ) {
        if ( isloaded ) {
            cudaMemcpy ( hPtr, movePtr(dPtr, numElements), Tbytes * ghostWidth,
                         cudaMemcpyDeviceToHost );
        }
        return movePtr(hPtr, numElements);
    }

    dim3 AgentsPartition::blockDim ( ) {
        return dims[ 0 ];
    }

    dim3 AgentsPartition::threadDim ( ) {
        return dims[ 1 ];
    }

    void AgentsPartition::setIdealDims ( ) {
        int numBlocks = ( numElements - 1 ) / BLOCK_SIZE + 1;
        dim3 blockDim ( numBlocks, 1, 1 );

        int nThr = ( numElements - 1 ) / numBlocks + 1;
        dim3 threadDim ( nThr, 1, 1 );

        dims[ 0 ] = blockDim;
        dims[ 1 ] = threadDim;
    }

    int AgentsPartition::getPlaceBytes ( ) {
        return Tbytes;
    }


    //void *hPtr; // this starts at the left ghost, and extends to the end of the right ghost
    //void *dPtr; // pointer to GPU data
    //int handle;         // User-defined identifier for this AgentsPartition
    //int rank; // the rank of this partition
    //int numElements;    // the number of agent elements in this AgentsPartition
    //int Tbytes; // sizeof(agent)
    //bool isloaded;
    //bool loadable;
    //int ghostWidth;
    //dim3 dims[ 2 ]; // 0 is blockdim, 1 is threaddim

}// mass namespace
