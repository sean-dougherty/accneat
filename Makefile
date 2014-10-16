	include Makefile.conf

INCLUDES=$(patsubst %,-I%,$(shell find src -type d))
SOURCES=$(shell find src -name "*.cpp")
OBJECTS=${SOURCES:src/%.cpp=obj/cpp/%.o}
DEPENDS=${OBJECTS:%.o=%.d}

LIBS=-lgomp
DEFINES=

ifdef ENABLE_CUDA
	CUDA_SOURCES=$(shell find src -name "*.cu")
	CUDA_OBJECTS=${CUDA_SOURCES:src/%.cu=obj/cu/%.o}
	LIBS+=-lcudart
	DEFINES+=-DENABLE_CUDA
endif

ifdef DEVMODE
	OPT=-O0
	#OPENMP=-fopenmp
	#PROFILE=-pg
	MISC_FLAGS=
	NVCC_FLAGS=-G -g
else
	OPT=-O3
	OPENMP=-fopenmp
	MISC_FLAGS=-Werror
endif

CC_FLAGS=-Wall ${DEFINES} ${MISC_FLAGS} ${PROFILE} ${INCLUDES} ${OPENMP} ${OPT} -c -std=c++11 -g -gdwarf-3

./neat: ${OBJECTS} ${CUDA_OBJECTS}
	g++ ${PROFILE} ${OBJECTS} ${CUDA_OBJECTS} ${PFM_LD_FLAGS} ${LIBS} -o $@

.PHONY: clean
clean:
	rm -rf obj
	rm -f ./neat
	rm -f src/util/std.h.gch

src/util/std.h.gch: src/util/std.h Makefile.conf Makefile
	g++ ${CC_FLAGS} $< -o $@

obj/cpp/%.o: src/%.cpp Makefile.conf Makefile src/util/std.h.gch
	@mkdir -p $(shell dirname $@)
	g++ ${CC_FLAGS} -MMD $< -o $@

obj/cu/%.o: src/%.cu src/%.h Makefile.conf Makefile
	@mkdir -p $(shell dirname $@)
	nvcc ${NVCC_FLAGS} -Xcompiler "${OPT}" -Isrc -c -arch=sm_13 --compiler-bindir ${PFM_NVCC_CCBIN} $< -o $@

-include ${DEPENDS}
