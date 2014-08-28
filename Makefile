SOURCES=$(wildcard src/*.cpp)
OBJECTS=${SOURCES:src/%.cpp=obj/%.o}
DEPENDS=${OBJECTS:%.o=%.d}

./neat: ${OBJECTS}
	g++ ${OBJECTS} -o $@

.PHONY: clean
clean:
	rm -rf obj
	rm -f ./neat

obj/%.o: src/%.cpp
	@mkdir -p obj
	g++ -MMD -c -std=c++11 $< -o $@

-include ${DEPENDS}
