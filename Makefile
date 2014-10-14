#ENABLE_CUDA=non-empty
#DEVMODE=non-empty

INCLUDES=$(patsubst %,-I%,$(shell find src -type d))
SOURCES=$(shell find src -name "*.cpp")
OBJECTS=${SOURCES:src/%.cpp=obj/cpp/%.o}
DEPENDS=${OBJECTS:%.o=%.d}

ifdef ENABLE_CUDA
	CUDA_SOURCES=$(shell find src -name "*.cu")
	CUDA_OBJECTS=${CUDA_SOURCES:src/%.cu=obj/cu/%.o}
endif



ifdef DEVMODE
	OPT=-O0
	OPENMP=-fopenmp
	#PROFILE=-pg
	MISC_FLAGS=
else
	OPT=-O2
	OPENMP=-fopenmp
	MISC_FLAGS=-Werror
endif

CC_FLAGS=-Wall ${MISC_FLAGS} ${PROFILE} ${INCLUDES} ${OPENMP} ${OPT} -c -std=c++11 -g -gdwarf-3

./neat: ${OBJECTS} ${CUDA_OBJECTS}
	g++ ${PROFILE} ${OBJECTS} ${CUDA_OBJECTS} -lgomp -lcudart -o $@

.PHONY: clean
clean:
	rm -rf obj
	rm -f ./neat
	rm -f src/util/std.h.gch

src/util/std.h.gch: src/util/std.h Makefile
	g++ ${CC_FLAGS} $< -o $@

obj/cpp/%.o: src/%.cpp Makefile src/util/std.h.gch
	@mkdir -p $(shell dirname $@)
	g++ ${CC_FLAGS} -MMD $< -o $@

obj/cu/%.o: src/%.cu src/%.h Makefile
	@mkdir -p $(shell dirname $@)
	nvcc -Isrc -c --compiler-bindir bin $< -o $@

-include ${DEPENDS}
