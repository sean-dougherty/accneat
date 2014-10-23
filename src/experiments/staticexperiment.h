#pragma once

#include "evaluatorexperiment.h"
#include "staticevaluator.h"

namespace NEAT {

    typedef std::function<std::vector<Test> ()> GetStaticTestsFunc;

    inline void create_static_experiment(const char *name,
                                         GetStaticTestsFunc get_tests) {

        auto create_evaluator =
            [get_tests] () {
            return create_static_evaluator(get_tests());
        };

        auto create_seeds = [get_tests] (rng_t rng_exp) {
            Step s = get_tests().front().steps.front();

            return 
            env->genome_manager->create_seed_generation(env->pop_size,
                                                        rng_exp,
                                                        1,
                                                        s.input.size(),
                                                        s.output.size(),
                                                        s.input.size());
        };

        //todo: This is wonky. Should maybe make an explicit static registry func?
        new EvaluatorExperiment(name, create_evaluator, create_seeds);
    }

}
