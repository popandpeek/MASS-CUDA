all: dirs app

dirs:
	mkdir -p obj/lib
	mkdir -p obj/test
	mkdir -p bin
	mkdir -p lib

app: objlib objtest
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -lcurand -L/usr/local/cuda/lib64 obj/test/*.o lib/mass_cuda.a -o bin/app
	#chmod +x bin/app

objlib:
	# Flag -c only compiles files but not links them
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/DataModel.cu -o obj/lib/DataModel.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/DeviceConfig.cu -o obj/lib/DeviceConfig.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -c src/Dispatcher.cu -o obj/lib/Dispatcher.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/Mass.cu -o obj/lib/Mass.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/Place.cu -o obj/lib/Place.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/Places.cu -o obj/lib/Places.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/PlacesModel.cu -o obj/lib/PlacesModel.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/cudaUtil.cu -o obj/lib/cudaUtil.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/Logger.cpp -o obj/lib/Logger.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/MassException.cpp -o obj/lib/MassException.o
	ar ru lib/mass_cuda.a obj/lib/*.o
	ranlib lib/mass_cuda.a

objtest: 
	# Flag -c only compiles files but not links them
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test/Timer.cpp -o obj/test/Timer.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test/Heat2d.cu -o obj/test/Heat2d.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test/Metal.cu -o obj/test/Metal.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test/MetalState.cu -o obj/test/MetalState.o
	nvcc -arch=sm_20 -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test/main.cu -o obj/test/main.o

clean:
	rm -rf obj
	rm -rf bin
	rm -rf lib
