#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "staticexperiment.h"
#include <assert.h>

using namespace NEAT;
using namespace std;

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
    StaticEvaluator::create_config(tests, config, configlen);
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
