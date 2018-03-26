# MASS CUDA README #

### What is this repository for? ###

* This repo is for MASS CUDA library use and development. It contains the MASS CUDA library and applications using it - Heat 2D, SugarScape and test application.

### How do I get set up? ###

* Compilation
    
    Library comes with provided Makefile, so to compile & link the library and test application go to the main folder and run the command: `make`

* Optimization
    
    To tailor the library settings to a particular application, set the appropriate application-specific parameters in the src/settings.h file.

    To optimize the kernel launch configuration for the particular GPU performance, the user can change the BLOCK_SIZE parameter in the src/sudaUtil.h file.


* Running

    To run the Heat2D app run the command: `./bin/heat2d`. The results of the execution will be saved in the text file called `perf_test_results.txt`.

    To run the SugarScape app run the command `./bin/sugarscape`. The results of the execution will be saved in the file called `sugar_scape_test_results.txt`.

    To run the test program tun the command`./bin/test`. The results will be saved in file `test_program_logs.txt`.
    


* Repo owner or admin

    The most recent active developer of the library to contact:
    Lisa Kosiachenko
    ekosyachenko at gmail dot com

    
* Other community or team contact

    Professor Munehiro Fukuda 
    mfukuda at u dot washington dot education domain abbreviation