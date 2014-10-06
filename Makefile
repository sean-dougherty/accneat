SOURCES=$(shell find src -name "*.cpp")
INCLUDES=-Iobj $(patsubst %,-I%,$(shell find src -type d))
OBJECTS=${SOURCES:src/%.cpp=obj/%.o}
DEPENDS=${OBJECTS:%.o=%.d}

#PROFILE=-pg
OPENMP=-fopenmp
OPT=-O2

CC_FLAGS=-Wall -Werror ${PROFILE} ${INCLUDES} ${OPENMP} ${OPT} -c -std=c++11 -g -gdwarf-3

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
