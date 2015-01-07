/**
 *  @file AllTests.h
 *  @author Nate Hart
 *
 *  @section LICENSE
 *  This is a file for use in Nate Hart's Thesis for the UW Bothell MSCSSE. All rights reserved.
 */

#ifndef ALLTESTS_H_
#define ALLTESTS_H_

class AllTests {
public:

	int failedTests;

	AllTests();
	virtual ~AllTests();

	bool proofOfConcept();

	bool runDispatcherTests();

	bool runMassTests();

	bool runDllClassTests();

	bool runPlacesTests();

	bool runPlacesPartitionTests();

	bool runAgentsTests();

	bool runAgentsPartitionTests();
};

#endif /* ALLTESTS_H_ */
