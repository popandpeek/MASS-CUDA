# README #

TODO add lots of detail.

### What is this repository for? ###

* This repo is for Nate Hart's thesis work. It contains the code for the MASS CUDA library.
* Version 0.1
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up

    1. Get ahold of the[ Nsight Eclipse IDE](http://www.nvidia.com/object/nsight.html), available from NVidia.
    1. Create an empty CUDA / C++ project. 
    2. Select "Empty Project" under the "shared" folder when given an option. Name this project "mass_cuda"
    3. Select "Separate compilation" as the Device linker mode
    4. Check the '3.0' box for PTX code
    5. Check the '3.0' and '3.5' box for GPU code
    6. Finish project creation
    7. Clone this repo to the directory for the new project

* Configuration
* Dependencies
* Database configuration
* How to run tests

    This project uses the [Boost Unit Testing Framework](http://www.boost.org/doc/libs/1_57_0/more/getting_started/unix-variants.html) for unit tests.
    1. Right click on the project, select 'Properties'
    2. Under Build > Settings > Tool Settings > NVCC Compiler > Includes, add 'path/to/boost_1_57_0/boost'
    3. Click OK. You can now include boost header files.

* Deployment instructions

    Compile for release and run the main function.

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact