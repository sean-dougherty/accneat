#include "std.h"

#include "evaluatorexperiment.h"
#include "genomemanager.h"
#include "maze.h"
#include "neat.h"
#include "rng.h"

using namespace NEAT;

static struct MazeInit {
    MazeInit() {

        auto create_evaluator =
            [] () {
            return create_maze_evaluator();
        };

        auto create_seeds = [] (rng_t rng_exp) {
            return 
            env->genome_manager->create_seed_generation(env->pop_size,
                                                        rng_exp,
                                                        1,
                                                        __sensor_N,
                                                        __output_N,
                                                        __sensor_N);
        };

        //todo: This is wonky. Should maybe make an explicit static registry func?
        new EvaluatorExperiment("maze", create_evaluator, create_seeds);
    }
} init;
