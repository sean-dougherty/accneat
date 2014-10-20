#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "batteryexperiment.h"
#include <assert.h>

using namespace NEAT;
using namespace std;

//------------------------------
//---
//--- UTIL
//---
//------------------------------
static const char *str(Test::Type t) {
    switch(t) {
    case Test::Training: return "Training";
    case Test::Fittest: return "Fittest";
    default: panic();
    }
}

static const char *str(Step::ErrType t) {
    switch(t) {
    case Step::Err_Delta: return "delta";
    case Step::Err_Binary: return "binary";
    default: panic();
    }
}

//------------------------------
//---
//--- CLASS Step
//---
//------------------------------
Step::Step(const vector<real_t> &input_,
           const vector<real_t> &output_,
           real_t weight_,
           ErrType err_type_)
    : input(input_)
    , output(output_)
    , weight(weight_)
    , err_type(err_type_) {

    if(err_type == Err_Binary) {
        for(auto x: output) {
            assert(x == 0.0 || x == 1.0);
        }
    }
}

real_t Step::process_output(Network &net) {
    switch(err_type) {
    case Err_Delta: {
        real_t result = 0.0;

        for(size_t i = 0; i < output.size(); i++) {
            real_t err = abs(net.get_output(i) - output[i]);
            if(err < 0.05) {
                err = 0.0;
            }
            result += err;
        }

        return result * weight;
    } break;
    case Err_Binary: {
        size_t result = 0;

        for(size_t i = 0; i < output.size(); i++) {
            if( int(net.get_output(i) + 0.5) != int(output[i]) ) {
                result++;
            }
        }

        return real_t(result) * weight;
    } break;
    default:
        panic();
    }
}

//------------------------------
//---
//--- CLASS Test
//---
//------------------------------
Test::Test(const string &name_,
           const vector<Step> &steps_,
           Type type_)
    : name(name_), steps(steps_), type(type_) {
}

real_t Test::process_output(Network &net, size_t istep) {
    return steps[istep].process_output(net);
}

//------------------------------
//---
//--- CLASS TestBattery
//---
//------------------------------
TestBattery::TestBattery(const std::vector<Test> &tests_) : tests(tests_) {
    population_err.resize(env->pop_size);
    
    for(size_t itest = 0; itest < tests.size(); itest++) {
        for(size_t istep = 0; istep < tests[itest].steps.size(); istep++) {
            test_steps.push_back({itest, istep});
        }
    }

    max_err = 0.0;
    for(Test &test: tests) {
        for(Step &step: test.steps) {
            max_err += step.weight * step.output.size();
        }
    }

    batch_sensors = env->network_manager->make_batch_sensors(tests[0].steps[0].input.size(),
                                                             test_steps.size());
    for(size_t istep = 0; istep < test_steps.size(); istep++) {
        TestStep tstep = test_steps[istep];
        batch_sensors->configure_step(istep,
                                      tests[tstep.itest].steps[tstep.istep].input,
                                      tstep.istep == 0);
    }
}

void TestBattery::process_output(Network &net, size_t istep) {
    if(istep == 0)
        population_err[net.population_index] = 0.0;

    TestStep &step = test_steps[istep];

    population_err[net.population_index] += tests[step.itest].process_output(net, step.istep);
}

OrganismEvaluation TestBattery::get_evaluation(Organism &org) {
    OrganismEvaluation result;
    result.error = population_err[org.population_index];
    result.fitness = 1.0 - result.error/max_err;
    return result;
}

//------------------------------
//---
//--- CLASS BatteryExperiment
//---
//------------------------------
void BatteryExperiment::init_experiment() {
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

    {
        map<Test::Type, vector<Test>> testmap;
        //Organize tests into batteries.
        for(Test &test: tests) {
            testmap[test.type].push_back(test);
        }
        assert( contains(testmap, Test::Training) );

        for(auto &kv: testmap) {
            batteries.emplace(kv.first, kv.second);
        }
    }

    cout << "=====================" << endl;
    cout << "===== BATTERIES =====" << endl;
    cout << "=====================" << endl;
    for(auto &kv: batteries) {
        cout << "---" << endl;
        cout << "---" << str(kv.first) << endl;
        cout << "---" << endl;

        for(Test &t: kv.second.tests) {
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
                printf(", err_type=%s", str(s.err_type));
                printf("\n");
            }
        }
    }
}

BatchSensors *BatteryExperiment::get_sensors() {
    return batteries.find(Test::Training)->second.batch_sensors.get();
}

void BatteryExperiment::process_output(class Network &net, size_t istep) {
    batteries.find(Test::Training)->second.process_output(net, istep);
}

OrganismEvaluation BatteryExperiment::evaluate(Organism &org) {
    return batteries.find(Test::Training)->second.get_evaluation(org);
}
