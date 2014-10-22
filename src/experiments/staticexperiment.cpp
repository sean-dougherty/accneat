#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "staticexperiment.h"
#include <assert.h>

using namespace NEAT;
using namespace std;

static void create_config(std::vector<Test> &tests,
                          __out const StaticConfig *&config_,
                          __out size_t &len_) {
    size_t nsteps = 0;
    real_t max_err = 0.0;

    for(Test &test: tests) {
        nsteps += test.steps.size();
        for(Step &step: test.steps) {
            max_err += step.weight * step.output.size();
        }
    }

    node_size_t ninputs = tests[0].steps[0].input.size();
    node_size_t noutputs = tests[0].steps[0].output.size();
    size_t len = StaticConfig::sizeof_buffer(nsteps, ninputs, noutputs);
    StaticConfig *config = (StaticConfig *)malloc(len);

    config->max_err = max_err;
    config->ninputs = ninputs;
    config->noutputs = noutputs;
    config->nsteps = nsteps;

    {
        size_t istep = 0;
        for(Test &t: tests) {
            bool first = true;
            for(Step &step: t.steps) {
                StaticConfig::StepParms *parms = config->parms(istep);
                parms->clear_noninput = first;
                parms->weight = step.weight;
                        
                for(node_size_t i = 0; i < ninputs; i++)
                    config->inputs(istep)[i] = step.input[i];

                for(node_size_t i = 0; i < noutputs; i++)
                    config->outputs(istep)[i] = step.output[i];

                first = false;
                istep++;
            }
        }
    }

    config_ = config;
    len_ = len;
}
//------------------------------
//---
//--- CLASS StaticExperiment
//---
//------------------------------
void StaticExperiment::init_experiment() {
    vector<Test> tests = create_tests();

    //Validate tests
    {
        assert(tests.size() > 0);
        assert(tests[0].steps.size() > 0);

        ninputs = tests[0].steps[0].input.size();
        assert(ninputs > 0);

        noutputs = tests[0].steps[0].output.size();
        assert(noutputs > 0);

        for(size_t i = 1; i < tests.size(); i++) {
            Test &test = tests[i];
            assert(test.steps.size() > 0);
            for(Step &step: tests[i].steps) {
                assert(step.input.size() == ninputs);
                assert(step.output.size() == noutputs);
            }
        }
    }

    //---
    //--- Config network executor
    //---
    const StaticEvaluator::Config *config;
    size_t configlen;
    create_config(tests, config, configlen);
    network_executor->configure(config, configlen);

    //---
    //--- Show tests
    //---
    cout << "=================" << endl;
    cout << "===== TESTS =====" << endl;
    cout << "=================" << endl;
    for(Test &t: tests) {
        printf("~~~ %s\n", t.name.c_str());
        for(Step s: t.steps) {
            for(real_t i: s.input) {
                printf("%1.3f ", i);
            }
            printf("| ");
            for(real_t o: s.output) {
                printf("%1.3f ", o);
            }
            printf(" ; weight=%f", s.weight);
            printf("\n");
        }
    }
}
