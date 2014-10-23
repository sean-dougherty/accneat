include Makefile.conf

INCLUDES=$(patsubst %,-I%,$(shell find src -type d))
SOURCES=$(shell find src -name "*.cpp")
CXX_SOURCES=$(shell find src -name "*.cxx")

OBJECTS=${SOURCES:src/%.cpp=obj/cpp/%.o}

LIBS=-lgomp
DEFINES=

ifeq (${ENABLE_CUDA}, true)
	CUDA_SOURCES=$(shell find src -name "*.cu")
	CUDA_OBJECTS=${CUDA_SOURCES:src/%.cu=obj/cu/%.o}
	CUDA_OBJECTS+=${CXX_SOURCES:src/%.cxx=obj/cu/cxx/%.o}
	LIBS+=-lcudart
	DEFINES+=-DENABLE_CUDA
else
	OBJECTS+=${CXX_SOURCES:src/%.cxx=obj/cpp/cxx/%.o}
endif

DEPENDS=${OBJECTS:%.o=%.d}

ifeq (${DEVMODE}, true)
	OPT=-O0
	#OPENMP=-fopenmp
	MISC_FLAGS=
	NVCC_FLAGS=-G -g
else
	OPT=-O3
	OPENMP=-fopenmp
	MISC_FLAGS=-Werror
endif

CC_FLAGS=-Wall ${DEFINES} ${MISC_FLAGS} ${PROFILE} ${INCLUDES} ${OPENMP} ${OPT} -c -g -gdwarf-3

./neat: ${OBJECTS} ${CUDA_OBJECTS}
	g++ ${PROFILE} ${OBJECTS} ${CUDA_OBJECTS} ${PFM_LD_FLAGS} ${LIBS} -o $@

.PHONY: clean
clean:
	rm -rf obj
	rm -f ./neat
	rm -f src/util/std.h.gch

src/util/std.h.gch: src/util/std.h Makefile.conf Makefile
	g++ ${CC_FLAGS} -std=c++11 $< -o $@

ifeq (${ENABLE_CUDA}, true)
obj/cu/cxx/%.o: src/%.cxx src/%.h Makefile.conf Makefile
	@mkdir -p $(shell dirname $@)
	nvcc -x cu -DENABLE_CUDA ${NVCC_FLAGS} -Xcompiler "${OPT} ${INCLUDES}" -c -arch=sm_13 --compiler-bindir ${PFM_NVCC_CCBIN} $< -o $@
else
obj/cpp/cxx/%.o: src/%.cxx Makefile.conf Makefile
	@mkdir -p $(shell dirname $@)
	g++ ${CC_FLAGS} -std=c++98 -MMD $< -o $@
endif

obj/cpp/%.o: src/%.cpp Makefile.conf Makefile src/util/std.h.gch
	@mkdir -p $(shell dirname $@)
	g++ ${CC_FLAGS} -std=c++11 -MMD $< -o $@

obj/cu/%.o: src/%.cu src/%.h Makefile.conf Makefile
	@mkdir -p $(shell dirname $@)
	nvcc -DENABLE_CUDA ${NVCC_FLAGS} -Xcompiler "${OPT}" -Isrc -c -arch=sm_13 --compiler-bindir ${PFM_NVCC_CCBIN} $< -o $@

-include ${DEPENDS}
