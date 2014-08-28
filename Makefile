SOURCES=$(wildcard src/*.cpp)
OBJECTS=${SOURCES:src/%.cpp=obj/%.o}
DEPENDS=${OBJECTS:%.o=%.d}

./neat: ${OBJECTS}
	g++ ${OBJECTS} -lgomp -o $@

.PHONY: clean
clean:
	rm -rf obj
	rm -f ./neat

obj/%.o: src/%.cpp Makefile
	@mkdir -p obj
	g++ -fopenmp -MMD -O3 -c -std=c++11 $< -o $@

-include ${DEPENDS}
