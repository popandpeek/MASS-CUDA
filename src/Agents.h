
#pragma once

#include <map>
#include <string>
#include <vector>

#include "Logger.h"

namespace mass {

// forward declarations
class Dispatcher;
class Agent;

class Agents {
    friend class Mass;

public:

    /**
     *  Creates a Agents object.
     *
     *  @param handle the unique identifier of this Agents collections
     */
    Agents(int handle, int nAgents, Dispatcher *d, int placesHandle);

    ~Agents();

    /**
     * Returns the number of agents present in this agents collection.
     * @return
     */
    int getNumAgents();

    /**
     *  Returns the handle associated with this Agents object that was set at construction.
     */
    int getHandle();

    /**
     *  Executes the given functionId on each Agent element within this Places.
     *
     *  @param functionId the function id passed to each Agent element
     */
    void callAll(int functionId);

    /**
     *  Executes the given functionId on each Agent element within this Places with
     *  the provided argument.
     *
     *  @param functionId the function id passed to each Agent element
     *  @param argument the argument to be passed to each Agent element
     *  @param argSize the size in bytes of the argument
     */
    void callAll(int functionId, void *argument, int argSize);

    /**
     *  Calls the function specified on all agent elements by passing argument[i]
     *  to place[i]'s function, and receives a value from it into (void *)[i] whose
     *  element size is retSize bytes.
     *
     *  @param functionId the function id passed to each Agent element
     *  @param arguments the arguments to be passed to each Agent element
     *  @param argSize the size in bytes of each argument element
     *  @param retSize the size in bytes of the return array element
     */
    // void *callAll(int functionId, void *arguments[], int argSize, int retSize);

     /*
     Executes agent termination / migration / spawning, which was initiated since the previous call to manageAll()
     */
    void manageAll();

    /**
     *  Returns an array of pointers to the Agent elements contained in this
     *  Agents object. This is an expensive operation since it requires memory
     *  transfer. This array should NOT be deleted.
     */
    Agent** getElements();

private:

    int handle;         // User-defined identifier for this Agents collection
    int placesHandle;   // Places collection associated with these Agents
    Dispatcher *dispatcher; // the GPU dispatcher

    int numElements;
    Agent **elemPtrs;

};

} /* namespace mass */
