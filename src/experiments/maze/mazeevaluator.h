#pragma once

namespace NEAT {

    enum sensor_t {
        sensor_right = 0,
        sensor_fwd = 1,
        sensor_left = 2,
        sensor_sound = 3,
        sensor_freq = 4,
        sensor_go = 5,
        __sensor_N = 6 
    };

    enum output_t {
        output_right = 0,
        output_left = 1,
        output_fwd = 2,
        __output_N = 3
    };

    class NetworkEvaluator *create_maze_evaluator();
}
