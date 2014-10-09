SOURCES=$(shell find src -name "*.cpp")
INCLUDES=-Iobj $(patsubst %,-I%,$(shell find src -type d))
OBJECTS=${SOURCES:src/%.cpp=obj/%.o}
DEPENDS=${OBJECTS:%.o=%.d}

#DEVMODE=non-empty

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

./neat: ${OBJECTS}
	g++ ${PROFILE} ${OBJECTS} -lgomp -o $@

.PHONY: clean
clean:
	rm -rf obj
	rm -f ./neat
	rm -f src/util/std.h.gch

src/util/std.h.gch: src/util/std.h Makefile
	g++ ${CC_FLAGS} $< -o $@

obj/%.o: src/%.cpp Makefile src/util/std.h.gch
	@mkdir -p $(shell dirname $@)
	g++ ${CC_FLAGS} -MMD $< -o $@

-include ${DEPENDS}
