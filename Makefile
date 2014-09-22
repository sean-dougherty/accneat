SOURCES=$(shell find src -name "*.cpp")
INCLUDES=$(patsubst %,-I%,$(shell find src -type d))
OBJECTS=${SOURCES:src/%.cpp=obj/%.o}
DEPENDS=${OBJECTS:%.o=%.d}

#PROFILE=-pg
OPENMP=-fopenmp
OPT=-O3

./neat: ${OBJECTS}
	g++ ${PROFILE} ${OBJECTS} -lgomp -o $@

.PHONY: clean
clean:
	rm -rf obj
	rm -f ./neat

obj/%.o: src/%.cpp Makefile
	@mkdir -p $(shell dirname $@)
	g++ -Wall ${PROFILE} ${INCLUDES} -MMD ${OPENMP} ${OPT} -c -std=c++11 -g -gdwarf-3 $< -o $@

-include ${DEPENDS}
