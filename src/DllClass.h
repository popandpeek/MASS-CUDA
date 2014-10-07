/*
 * DllClass.h
 *
 *  Created on: Oct 7, 2014
 *      Author: natehart
 */

#pragma once

#include <string>
#include <dlfcn.h> // dlopen dlsym dlclose

namespace mass {

class DllClass {
 public:
//  void *stub;
//  instantiate_t *instantiate;
//  destroy_t *destroy;
//
  DllClass( std::string className );
};

} /* namespace mass */
