#pragma once

#include "evaluatorexperiment.h"

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

    struct StaticConfig {
        struct StepParms {
            union {
                bool clear_noninput;
                real_t __padding;
            };
            real_t weight;
        };
        real_t max_err;
        node_size_t ninputs;
        node_size_t noutputs;
        size_t nsteps;
        uchar payload[];

        static size_t sizeof_step(node_size_t ninputs, node_size_t noutputs) {
            return sizeof(StepParms) + sizeof(real_t) * (ninputs+noutputs);
        }
        static size_t sizeof_buffer(size_t nsteps, node_size_t ninputs, node_size_t noutputs) {
            return sizeof(StaticConfig) + nsteps * sizeof_step(ninputs, noutputs);
        }

        size_t offset_step(size_t istep) const {
            return sizeof_step(ninputs, noutputs) * istep;
        }

        StepParms *parms(size_t istep) const {
            return (StepParms *)(payload + offset_step(istep));
        }
        real_t *inputs(size_t istep) const {
            return (real_t *)(parms(istep) + 1);
        }
        real_t *outputs(size_t istep) const {
            return inputs(istep) + ninputs;
        }
    };

    struct StaticEvaluator {

        typedef StaticConfig Config;

        const Config *config;
        real_t errorsum;

        StaticEvaluator(const Config *config_)
        : config(config_) {
            errorsum = 0.0;
        }

        bool complete(size_t istep) {
            return istep >= config->nsteps;
        }

        bool clear_noninput(size_t istep) {
            return config->parms(istep)->clear_noninput;
        }

        real_t get_sensor(size_t istep,
                          size_t sensor_index) {
            return config->inputs(istep)[sensor_index];
        }

        void evaluate(size_t istep, real_t *actual) {
            real_t *expected = config->outputs(istep);
            real_t result = 0.0;

            for(size_t i = 0; i < config->noutputs; i++) {
                real_t err = actual[i] - expected[i];
                if(err < 0) err *= -1;
                if(err < 0.05) {
                    err = 0.0;
                }
                result += err;
            }

            errorsum += result * config->parms(istep)->weight;
        }

        OrganismEvaluation result() {
            OrganismEvaluation eval;
            eval.error = errorsum;
            eval.fitness = 1.0 - errorsum/config->max_err;
            return eval;
        }
    };

    class StaticExperiment : public EvaluatorExperiment<StaticEvaluator> {
    public:
        StaticExperiment(const char *name) : EvaluatorExperiment<StaticEvaluator>(name) {}
        virtual ~StaticExperiment() {}

        virtual void init_experiment() override;
        virtual std::vector<Test> create_tests() = 0;
    };

}
