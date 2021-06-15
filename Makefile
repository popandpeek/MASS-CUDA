all: dirs braingrid heat2d sugarscape test

dirs:
	mkdir -p obj/lib
	mkdir -p obj/heat2d
	mkdir -p obj/sugarscape
	mkdir -p obj/braingrid
	mkdir -p obj/test
	mkdir -p bin
	mkdir -p lib

heat2d: objlib objheat2d
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -Xlinker -lgomp -lineinfo -lcurand -m64 -L/usr/local/cuda/lib64 obj/heat2d/*.o lib/mass_cuda.a -o bin/heat2d

sugarscape: objlib objsugarscape
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -Xlinker -lgomp -lineinfo -lcurand -m64 -L/usr/local/cuda/lib64 obj/sugarscape/*.o lib/mass_cuda.a -o bin/sugarscape

braingrid: objlib objbraingrid
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -Xlinker -lgomp -lineinfo -lcurand -m64 -L/usr/local/cuda/lib64 obj/braingrid/*.o lib/mass_cuda.a -o bin/braingrid

test: objlib objheat2d objtest
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -Xlinker -lgomp -lineinfo -lcurand -m64 -L/usr/local/cuda/lib64 obj/heat2d/Timer.o obj/heat2d/Heat2d.o obj/heat2d/Metal.o obj/heat2d/MetalState.o obj/sugarscape/SugarScape.o obj/sugarscape/SugarPlace.o obj/sugarscape/SugarPlaceState.o obj/sugarscape/Ant.o obj/sugarscape/AntState.o obj/test/TestPlace.o obj/test/TestAgent.o obj/test/test.o lib/mass_cuda.a -o bin/test

objlib:
	# Flag -c only compiles files but not links them
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 src/DataModel.cu -o obj/lib/DataModel.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -Xptxas -v -c -Xcompiler -fopenmp,-lgomp -m64 src/DeviceConfig.cu -o obj/lib/DeviceConfig.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -Xptxas -v -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -c -Xcompiler -fopenmp,-lgomp src/Dispatcher.cu -o obj/lib/Dispatcher.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -m64 -c -Xcompiler -fopenmp,-lgomp src/Mass.cu -o obj/lib/Mass.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -m64 -c -Xcompiler -fopenmp,-lgomp src/Agent.cu -o obj/lib/Agent.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -m64 -c -Xcompiler -fopenmp,-lgomp src/Place.cu -o obj/lib/Place.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -m64 -c -Xcompiler -fopenmp,-lgomp src/Places.cu -o obj/lib/Places.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -m64 -c -Xcompiler -fopenmp,-lgomp src/Agents.cu -o obj/lib/Agents.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -m64 -c -Xcompiler -fopenmp,-lgomp src/AgentsModel.cu -o obj/lib/AgentsModel.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -m64 -c -Xcompiler -fopenmp,-lgomp src/PlacesModel.cu -o obj/lib/PlacesModel.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -m64 -c -Xcompiler -fopenmp,-lgomp src/cudaUtil.cu -o obj/lib/cudaUtil.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -m64 -c -Xcompiler -fopenmp,-lgomp src/Logger.cpp -o obj/lib/Logger.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -m64 -c -Xcompiler -fopenmp,-lgomp src/MassException.cpp -o obj/lib/MassException.o
	ar ru lib/mass_cuda.a obj/lib/*.o
	ranlib lib/mass_cuda.a

objheat2d: 
	# Flag -c only compiles files but not links them
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -std=c++11 -c -Xcompiler -fopenmp,-lgomp heat2d/Timer.cpp -o obj/heat2d/Timer.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -std=c++11 -c -Xcompiler -fopenmp,-lgomp heat2d/Heat2d.cu -o obj/heat2d/Heat2d.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -std=c++11 -c -Xcompiler -fopenmp,-lgomp heat2d/Metal.cu -o obj/heat2d/Metal.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -std=c++11 -c -Xcompiler -fopenmp,-lgomp heat2d/MetalState.cu -o obj/heat2d/MetalState.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -std=c++11 -c -Xcompiler -fopenmp,-lgomp heat2d/main.cu -o obj/heat2d/main.o

objsugarscape:
	# Flag -c only compiles files but not links them
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 sugarscape/Timer.cpp -o obj/sugarscape/Timer.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 sugarscape/SugarScape.cu -o obj/sugarscape/SugarScape.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 sugarscape/SugarPlace.cu -o obj/sugarscape/SugarPlace.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 sugarscape/SugarPlaceState.cu -o obj/sugarscape/SugarPlaceState.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 sugarscape/Ant.cu -o obj/sugarscape/Ant.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 sugarscape/AntState.cu -o obj/sugarscape/AntState.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 sugarscape/main.cu -o obj/sugarscape/main.o

objbraingrid:
	# Flag -c only compiles files but not links them
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 braingrid/Timer.cpp -o obj/braingrid/Timer.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 braingrid/BrainGrid.cu -o obj/braingrid/BrainGrid.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 braingrid/NeuronPlace.cu -o obj/braingrid/NeuronPlace.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 braingrid/NeuronPlaceState.cu -o obj/braingrid/NeuronPlaceState.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 braingrid/GrowingEnd.cu -o obj/braingrid/GrowingEnd.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 braingrid/GrowingEndState.cu -o obj/braingrid/GrowingEndState.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -c -Xcompiler -fopenmp,-lgomp -m64 braingrid/main.cu -o obj/braingrid/main.o

objtest:
	# Flag -c only compiles files but not links them
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -std=c++11 -c -Xcompiler -fopenmp,-lgomp test/TestPlace.cu -o obj/test/TestPlace.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -std=c++11 -c -Xcompiler -fopenmp,-lgomp test/TestAgent.cu -o obj/test/TestAgent.o
	nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_75,code=sm_75 -rdc=true -std=c++11 -c -Xcompiler -fopenmp,-lgomp test/test.cu -o obj/test/test.o

clean:
	rm -rf obj
	rm -rf bin
	rm -rf lib
