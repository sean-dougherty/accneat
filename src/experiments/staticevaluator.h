#pragma once

#include "neattypes.h"

namespace NEAT {

    // Specifies a set of input activations and an expected set of output activations.
    struct Step {
        std::vector<real_t> input;
        std::vector<real_t> output;
        real_t weight;

        Step(const std::vector<real_t> &input_,
             const std::vector<real_t> &output_,
             real_t weight_ = 1.0)
        : input(input_)
        , output(output_)
        , weight(weight_) {
        }
    };

    // A set of Steps for which the neural net state is expected to begin in its default
    // state.
    struct Test {
        std::string name;
        std::vector<Step> steps;

        Test(const std::string &name_,
             const std::vector<Step> &steps_)
        : name(name_), steps(steps_) {
        }
        Test(const std::vector<Step> &steps_) : Test("", steps_) {}
    };

    extern class NetworkEvaluator *create_static_evaluator(const std::vector<Test> &tests);

}
