include Makefile.conf

CC_CUDA=nvcc -DENABLE_CUDA ${NVCC_FLAGS} -arch=sm_13 --compiler-bindir ${PFM_NVCC_CCBIN}

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
DEPENDS+=${CUDA_OBJECTS:%.o=%.d}

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
.PHONY: clean default

default: ./neat

clean:
	rm -rf obj
	rm -f ./neat
	rm -f src/util/std.h.gch

./neat: ${OBJECTS} ${CUDA_OBJECTS}
	g++ ${PROFILE} ${OBJECTS} ${CUDA_OBJECTS} ${PFM_LD_FLAGS} ${LIBS} -o $@


src/util/std.h.gch: src/util/std.h Makefile.conf Makefile
	g++ ${CC_FLAGS} -std=c++11 $< -o $@

ifeq (${ENABLE_CUDA}, true)

obj/cu/cxx/%.o: src/%.cxx Makefile.conf Makefile
	@mkdir -p $(shell dirname $@)
	${CC_CUDA} -c -x cu -Xcompiler "${OPT} ${INCLUDES}" $< -o $@

obj/cu/cxx/%.d: src/%.cxx
	@mkdir -p $(dir $@)
	@${CC_CUDA} -M -x cu -Xcompiler "${OPT} ${INCLUDES}" $< > $@.tmp
	@cat $@.tmp | sed 's,.*\.o[[:space:]]*:,$@:,g' | sed 's,\.d,\.o,' > $@
	@rm $@.tmp

obj/cu/%.o: src/%.cu Makefile.conf Makefile
	@mkdir -p $(shell dirname $@)
	${CC_CUDA} -c -Xcompiler "${OPT}" -Isrc $< -o $@

obj/cu/%.d: src/%.cu
	@mkdir -p $(dir $@)
	@${CC_CUDA} -M -Xcompiler "${OPT}" -Isrc $< > $@.tmp
	@cat $@.tmp | sed 's,.*\.o[[:space:]]*:,$@:,g' | sed 's,\.d,\.o,' > $@
	@rm $@.tmp

else
obj/cpp/cxx/%.o: src/%.cxx Makefile.conf Makefile
	@mkdir -p $(shell dirname $@)
	g++ ${CC_FLAGS} -std=c++98 -MMD $< -o $@
endif

obj/cpp/%.o: src/%.cpp Makefile.conf Makefile src/util/std.h.gch
	@mkdir -p $(shell dirname $@)
	g++ ${CC_FLAGS} -std=c++11 -MMD $< -o $@

-include ${DEPENDS}
