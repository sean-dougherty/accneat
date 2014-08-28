build/neat: build/Makefile
	cd build
	make

build/Makefile:
	mkdir -p build
	cmake ..
