
#ifndef AGENTS_H 
#define AGENTS_H

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
    Agents(int handle, Dispatcher *d, int placesHandle);

    ~Agents();

    /**
     * Returns the number of alive agents present in this agents collection.
     * @return
     */
    int getNumAgents();


    int* getNumAgentsInstantiated();
    
    /*
    Returns the number of all agent objects present in this agents collection (some can be terminated).
    */
    int getNumAgentObjects();

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

    /*
     Executes agent termination / migration / spawning, which was initiated since the previous call to manageAll()
     */
    void manageAll();

    void manageAllSpawnFirst();
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

    int* numAgents;
    std::vector<Agent**> elemPtrs;

};

} /* namespace mass */
#endif