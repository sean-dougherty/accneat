#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "networkexecutor.h"
#include "staticevaluator.h"

#include <assert.h>

using namespace NEAT;
using namespace std;

//---
//--- CLASS Config
//---
//--- A representation of the experiment that can either be used by the CPU in
//--- system memory or be pushed down to the GPU. That is, this must be valid
//--- Cuda code.
//---
struct Config {
    //todo: make a Step class that has an activations[]
    struct StepParms {
        //todo: shouldn't need __padding. compiler should align weight properly.
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
    uchar steps[];

    __net_eval_decl static size_t sizeof_step(node_size_t ninputs, node_size_t noutputs) {
        return sizeof(StepParms) + sizeof(real_t) * (ninputs+noutputs);
    }
    __net_eval_decl static size_t sizeof_buffer(size_t nsteps, node_size_t ninputs, node_size_t noutputs) {
        return sizeof(Config) + nsteps * sizeof_step(ninputs, noutputs);
    }

    __net_eval_decl size_t offset_step(size_t istep) const {
        return sizeof_step(ninputs, noutputs) * istep;
    }

    __net_eval_decl StepParms *parms(size_t istep) const {
        return (StepParms *)(steps + offset_step(istep));
    }
    __net_eval_decl real_t *inputs(size_t istep) const {
        return (real_t *)(parms(istep) + 1);
    }
    __net_eval_decl real_t *outputs(size_t istep) const {
        return inputs(istep) + ninputs;
    }
};

//---
//--- CLASS Evaluator
//---
//--- Manages a Network's sensors and evaluates a Network's output based on
//--- the Config. Will be used by both CPU and GPU.
//---
struct Evaluator {
    typedef ::Config Config;

    const Config *config;
    real_t errorsum;

    __net_eval_decl Evaluator(const Config *config_)
    : config(config_) {
        errorsum = 0.0;
    }

    __net_eval_decl bool complete(size_t istep) {
        return istep >= config->nsteps;
    }

    __net_eval_decl bool clear_noninput(size_t istep) {
        return config->parms(istep)->clear_noninput;
    }

    __net_eval_decl real_t get_sensor(size_t istep,
                      size_t sensor_index) {
        return config->inputs(istep)[sensor_index];
    }

    __net_eval_decl void evaluate(size_t istep, real_t *actual) {
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

    __net_eval_decl OrganismEvaluation result() {
        OrganismEvaluation eval;
        eval.error = errorsum;
        eval.fitness = 1.0 - errorsum/config->max_err;
        return eval;
    }
};

//---
//--- FUNC create_config
//---
//--- Convert convenient experiment declaration into Config encoding.
//---
static void create_config(const std::vector<Test> &tests,
                          __out Config *&config_,
                          __out size_t &len_) {
    //Validate tests
    {
        assert(tests.size() > 0);
        assert(tests[0].steps.size() > 0);

        node_size_t ninputs = tests[0].steps[0].input.size();
        assert(ninputs > 0);

        node_size_t noutputs = tests[0].steps[0].output.size();
        assert(noutputs > 0);

        for(size_t i = 1; i < tests.size(); i++) {
            const Test &test = tests[i];
            assert(test.steps.size() > 0);
            for(size_t j = 0; j < test.steps.size(); j++) {
                const Step &step = test.steps[j];
                assert(step.input.size() == ninputs);
                assert(step.output.size() == noutputs);
            }
        }
    }

    size_t nsteps = 0;
    real_t max_err = 0.0;

    for(size_t i = 0; i < tests.size(); i++) {
        const Test &test = tests[i];
        nsteps += test.steps.size();
        for(size_t j = 0; j < test.steps.size(); j++) {
            const Step &step = test.steps[j];
            max_err += step.weight * step.output.size();
        }
    }

    node_size_t ninputs = tests[0].steps[0].input.size();
    node_size_t noutputs = tests[0].steps[0].output.size();
    size_t len = Config::sizeof_buffer(nsteps, ninputs, noutputs);
    Config *config = (Config *)malloc(len);

    config->max_err = max_err;
    config->ninputs = ninputs;
    config->noutputs = noutputs;
    config->nsteps = nsteps;

    {
        size_t istep = 0;
        for(size_t i = 0; i < tests.size(); i++) {
            const Test &t = tests[i];
            for(size_t j = 0; j < t.steps.size(); j++) {
                const Step &step = t.steps[j];
                Config::StepParms *parms = config->parms(istep);
                parms->clear_noninput = (j == 0);
                parms->weight = step.weight;
                        
                for(node_size_t i = 0; i < ninputs; i++)
                    config->inputs(istep)[i] = step.input[i];

                for(node_size_t i = 0; i < noutputs; i++)
                    config->outputs(istep)[i] = step.output[i];

                istep++;
            }
        }
    }

    config_ = config;
    len_ = len;

    //---
    //--- Show tests
    //---
    cout << "=================" << endl;
    cout << "===== TESTS =====" << endl;
    cout << "=================" << endl;
    for(size_t i = 0; i < tests.size(); i++) {
        const Test &t = tests[i];
        printf("~~~ %s\n", t.name.c_str());
        for(size_t j = 0; j < t.steps.size(); j++) {
            const Step s = t.steps[j];
            for(size_t ii = 0; ii < s.input.size(); ii++) {
                printf("%1.3f ", s.input[ii]);
            }
            printf("| ");
            for(size_t ii = 0; ii < s.output.size(); ii++) {
                printf("%1.3f ", s.output[ii]);
            }
            printf(" ; weight=%f", s.weight);
            printf("\n");
        }
    }
}

//---
//--- CLASS StaticNetworkEvaluator
//---
//--- Implementation of the NetworkEvaluator interface that serves to hide the
//--- Cuda code from the rest of the (C++11) system.
//---
class StaticNetworkEvaluator : public NetworkEvaluator {
    NetworkExecutor<Evaluator> *executor;
public:
    StaticNetworkEvaluator(const vector<Test> &tests) {
        executor = NetworkExecutor<Evaluator>::create();

        Evaluator::Config *config;
        size_t configlen;
        create_config(tests, config, configlen);
        executor->configure(config, configlen);
        free(config);
    }

    ~StaticNetworkEvaluator() {
        delete executor;
    }

    virtual void execute(class Network **nets_,
                         class OrganismEvaluation *results,
                         size_t nnets) {
        executor->execute(nets_, results, nnets);
    }

};

namespace NEAT {
    NetworkEvaluator *create_static_evaluator(const vector<Test> &tests) {
        return new StaticNetworkEvaluator(tests);
    }
}
