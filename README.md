# Introduction

## What is AccNEAT?

AccNEAT (Accelerated NEAT) is a fork of
[Kenneth Stanley's NEAT project](http://www.cs.ucf.edu/~kstanley/neat.html).
It is primarily concerned with reducing the amount of time required by NEAT to
evolve solutions to difficult problems. It is typically able to reduce the time
required by at least an order of magnitude, and for difficult problems as much
as three orders of magnitude. This acceleration of NEAT is accomplished via the
following strategies:

* Optimizations that improve the original implementation of NEAT without
altering its genetic/evolutionary algorithm (e.g. using datatstructures that
reduce CPU cache misses and search algorithms that require *O(log N)* instead
of *O(N)*).

* Take advantage of parallel hardware (i.e. multicore CPUs and GPUs).

* Additional genetic operators (e.g. node delete mutation) and search strategies
(e.g. a phased search technique similar to that
[described by Colin Green](http://sharpneat.sourceforge.net/phasedsearch.html)
) that serve to
prevent explosive growth in complexity of evolved networks.

Performance gains are most dramatic for very difficult problems, but benefits
can also be seen for small, simple experiments, like evolving a network that can
solve XOR. The following table shows the amount of time required to conduct
100 experiments in which a solution for XOR is successfully found, where
a different random starting population is used for each experiment:

**Table 1: Time required to pass 100 XOR experiments**

| Configuration         | Time        |
| --------------------- | -----------:|
| NEAT                  | 295 seconds |
| AccNEAT, 1 CPU core   |  66 seconds |
| AccNEAT, 2 CPU cores  |  39 seconds |
| AccNEAT, 4 CPU cores  |  24 seconds |
| AccNEAT, 8 CPU cores  |  16 seconds |
| AccNEAT, 12 CPU cores |  13 seconds |
| AccNEAT, GPU          |  13 seconds |

*Note: All configurations used the "complexify" search algorithm and a population
size of 10,000. The CPU was an Intel Xeon X5660 running at 2.8 GHz, and the
operating system was CentOS 6.5.*

The *seq-1bit-4el* experiment (provided with AccNEAT) is considerably more difficult
than XOR, making it a better showcase for AccNEAT's improved search algorithm and
parallelism. The following table shows how much progress AccNEAT is able to make in
30 minutes of execution with a variety of search and processor configurations. Note 
that the "complexify" search is the algorithm used by the original NEAT implementation.

**Table 2: Generations processed and fitness achieved within 30 minutes wall time**

| Configuration         | Search Algorithm | Generation | Fitness *(1.0 = perfect)* |
| ----------------------|------------------|-----------:|--------------------------:| 
| AccNEAT, 12 CPU cores | complexify       |      1,112 |                  0.878473 |
| AccNEAT, GPU          | complexify       |      1,532 |                  0.883114 |
| AccNEAT, 12 CPU cores | phased           |      2,175 |                  0.919856 |
| AccNEAT, GPU          | phased           |      4,978 |                  0.932065 |

What this table fails to illustrate is that the time required to compute additional
generations with the complexify search increases roughly exponentially, while the
phased search increases roughly linearly. In addition, the amount of RAM required by
the complexify search will easily exceed 16 GB after a couple days of execution, and it
will continue to require more with each passing generation. In other words, lengthy
executions of large populations are not technically feasible with the complexify search.

## What is the status of AccNEAT?

As of October 2014, AccNEAT is under active development.

If you want to use AccNEAT, you will need to download the source code and create your
experiments within the source tree. The structure of the project will hopefully be
updated in the near future such that AccNEAT is a library, allowing users to conveniently
develop their own experiments outside the AccNEAT source tree.

## System Requirements

* AccNEAT is currently only used on Linux (Xubuntu 14.04 and CentOS 6.5). It shouldn't be
too painful to run on other POSIX systems, but the build system is not designed ideally
for portability.

* C++ compiler with full support for C++11 standard (e.g. GCC 4.9)

* NVCC and an Nvidia graphics card if you want to use a GPU accelerator. The Cuda code
is currently written to support 1.3 compute capability devices. The only NVCC version
that has been tried is 6.0, but an earlier version may work as well. Note that NVCC won't
work with GCC 4.9, so you'll need to have an older version of GCC as well! GCC 4.1 and
4.4 have worked fine.

## Installing/Building

### Download the source:

```
git clone https://github.com/sean-dougherty/accneat.git
```

### Configure:

```
cd accneat
./configure
```

This is not a proper configure script. It just makes a default *Makefile.conf*, which is
in turn included by *Makefile*. You can modify the contents of *Makefile.conf* to enable
GPU support (set ENABLE_CUDA=true) or to enable the debug build (DEVMODE=true). You may
also use it for platform-specific settings. See Makefile.xubuntu and Makefile.maxwell,
which are versions of Makefile.conf that I use on my Xubuntu laptop and on a CentOS cluster.

### Build:

```
make
```

## Running experiments

Experiments are executed via the *./neat* command. Executing with no arguments will provide
a usage message:

```
usage: neat [OPTIONS]... experiment_name

experiment names: cfg-XSX, foo, lessthan, regex-XYXY, regex-aba, regex-aba-2bit, seq-1bit-2el, seq-1bit-3el, seq-1bit-4el, xor

OPTIONS
  -f                   Force deletion of any data from previous run.
  -c num_experiments   (default=1)
  -r RNG_seed          (default=1)
  -n population_size   (default=1000)
  -x max_generations   (default=10000)
  -s search_type       {phased, blended, complexify} (default=phased)
```

So, to run the XOR experiment 10 times with a population size of 5,000, and using the complexify search, you would type:

```
./neat -c 10 -n 5000 -s complexify xor
```

Results will be written to directories named *./experiment_i*. Note that neat will refuse to
run if ./experiment_* directories already exist, unless the -f option is specified, which will
delete the old directories.

## Making your own experiments

For an example of how to make your own experiment, look at *src/experiments/xor.cpp*, which
shows a simple declaration of input and output. For an example of a more complicated setup,
see *src/experiments/regex.cpp*. Simply put your source file in the *src/experiments* directory,
and it should be automatically built and will be available from the command-line tool.
