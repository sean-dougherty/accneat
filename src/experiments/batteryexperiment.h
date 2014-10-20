#pragma once

#include "experiment.h"

namespace NEAT {

    // Specifies a set of input activations and an expected set of output activations.
    struct Step {
        enum ErrType {
            Err_Delta,
            Err_Binary
        };

        std::vector<real_t> input;
        std::vector<real_t> output;
        real_t weight;
        ErrType err_type;

        Step(const std::vector<real_t> &input_,
             const std::vector<real_t> &output_,
             real_t weight_ = 1.0,
             ErrType err_type_ = Err_Delta);

        real_t process_output(Network &net);
    };

    // A set of Steps for which the neural net state is expected to begin in its default
    // state.
    struct Test {
        enum Type {
            Training,
            Fittest
        };

        std::string name;
        std::vector<Step> steps;
        Type type;

        Test(const std::string &name_,
             const std::vector<Step> &steps_,
             Type type_ = Training);

        Test(const std::vector<Step> &steps_, Type type_ = Training)
        : Test("", steps_, type_) {
        }

        real_t process_output(Network &net, size_t istep);
    };

    // A set of Tests that are all of the same type (e.g. Training).
    struct TestBattery {
        TestBattery(const std::vector<Test> &tests_);

        void process_output(Network &net, size_t istep);
        OrganismEvaluation get_evaluation(Organism &org);

        //void show_report(Organism &org);

        std::vector<Test> tests;
        real_t max_err;
        struct TestStep {
            size_t itest;
            size_t istep;
        };
        std::vector<TestStep> test_steps;
        std::vector<real_t> population_err;
        std::unique_ptr<BatchSensors> batch_sensors;
    };

    class BatteryExperiment : public Experiment {
    public:
        BatteryExperiment(const char *name) : Experiment(name) {}
        virtual ~BatteryExperiment() {}

        virtual void init_experiment() override;
        virtual std::vector<Test> create_tests() = 0;

        virtual BatchSensors *get_sensors() override;
        virtual void process_output(class Network &net, size_t istep) override;
        virtual OrganismEvaluation evaluate(Organism &org) override;

        std::map<Test::Type, TestBattery> batteries;
    };

}
