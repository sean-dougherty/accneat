SOURCES=$(wildcard src/*.cpp)
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
	@mkdir -p obj
	g++ -Wall ${PROFILE} -MMD ${OPENMP} ${OPT} -c -std=c++11 -g -gdwarf-3 $< -o $@

-include ${DEPENDS}
