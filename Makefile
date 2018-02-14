all: dirs app appagents test

dirs:
	mkdir -p obj/lib
	mkdir -p obj/test
	mkdir -p obj/test_agents
	mkdir -p obj/test_agent_spawn
	mkdir -p bin
	mkdir -p lib

app: objlib objtest
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -lcurand -L/usr/local/cuda/lib64 obj/test/Timer.o obj/test/Heat2d.o obj/test/Metal.o obj/test/MetalState.o obj/test/main.o lib/mass_cuda.a -o bin/app

test: objlib objtest objtestagentspawn
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -lcurand -L/usr/local/cuda/lib64 obj/test/Timer.o obj/test/Heat2d.o obj/test/Metal.o obj/test/MetalState.o obj/test_agents/SugarScape.o obj/test_agents/SugarPlace.o obj/test_agents/SugarPlaceState.o obj/test_agents/Ant.o obj/test_agents/AntState.o obj/test_agent_spawn/TestPlace.o obj/test_agent_spawn/TestAgent.o obj/test/test.o lib/mass_cuda.a -o bin/test

appagents: objlib objtestagents
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -lcurand -L/usr/local/cuda/lib64 obj/test_agents/*.o lib/mass_cuda.a -o bin/appagents

objlib:
	# Flag -c only compiles files but not links them
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/DataModel.cu -o obj/lib/DataModel.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/DeviceConfig.cu -o obj/lib/DeviceConfig.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -c src/Dispatcher.cu -o obj/lib/Dispatcher.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/Mass.cu -o obj/lib/Mass.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/Agent.cu -o obj/lib/Agent.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/Place.cu -o obj/lib/Place.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/Places.cu -o obj/lib/Places.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/Agents.cu -o obj/lib/Agents.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/AgentsModel.cu -o obj/lib/AgentsModel.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/PlacesModel.cu -o obj/lib/PlacesModel.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/cudaUtil.cu -o obj/lib/cudaUtil.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/Logger.cpp -o obj/lib/Logger.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c src/MassException.cpp -o obj/lib/MassException.o
	ar ru lib/mass_cuda.a obj/lib/*.o
	ranlib lib/mass_cuda.a

objtest: 
	# Flag -c only compiles files but not links them
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test/Timer.cpp -o obj/test/Timer.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test/Heat2d.cu -o obj/test/Heat2d.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test/Metal.cu -o obj/test/Metal.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test/MetalState.cu -o obj/test/MetalState.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test/main.cu -o obj/test/main.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test/test.cu -o obj/test/test.o

objtestagents:
	# Flag -c only compiles files but not links them
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test_agents/Timer.cpp -o obj/test_agents/Timer.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test_agents/SugarScape.cu -o obj/test_agents/SugarScape.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test_agents/SugarPlace.cu -o obj/test_agents/SugarPlace.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test_agents/SugarPlaceState.cu -o obj/test_agents/SugarPlaceState.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test_agents/Ant.cu -o obj/test_agents/Ant.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test_agents/AntState.cu -o obj/test_agents/AntState.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test_agents/main.cu -o obj/test_agents/main.o

objtestagentspawn:
	# Flag -c only compiles files but not links them
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test_agent_spawn/TestPlace.cu -o obj/test_agent_spawn/TestPlace.o
	nvcc -Wno-deprecated-gpu-targets -rdc=true -std=c++11 -c test_agent_spawn/TestAgent.cu -o obj/test_agent_spawn/TestAgent.o

clean:
	rm -rf obj
	rm -rf bin
	rm -rf lib
