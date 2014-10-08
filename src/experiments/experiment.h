#include "neat.h"
#include "organism.h"
#include "util.h"

namespace NEAT {

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

        real_t evaluate(class Organism &);
    };

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

        void prepare(Organism &org, size_t istep);
        real_t evaluate(Organism &org, size_t istep);
    };

    struct TestBattery {
        std::vector<Test> tests;
        real_t max_err = 0.0;
        struct TestStep {
            size_t itest;
            size_t istep;
        };
        std::vector<TestStep> test_steps;
        std::vector<real_t> population_err;

        TestBattery();

        void add(const Test &test);

        struct EvalResult {
            real_t fitness;
            real_t error;
        };
        
        void reset();

        bool prepare_step(class Organism &org, size_t istep);
        void evaluate_step(class Organism &org, size_t istep);
        EvalResult get_evaluation(class Organism &org);

        //void show_report(class Organism &org);
    };

    class Experiment {
    public:
        static Experiment *get(const char *name);
        static std::vector<std::string> get_names();

    private:
        static std::map<std::string, Experiment*> *experiments;

    public:
        virtual ~Experiment();

        void init();
        void run(class rng_t &rng, int gens);

    protected:
        Experiment(const char *name);
        virtual void init_env();
        virtual std::vector<Test> create_tests() = 0;
        virtual bool is_success(class Organism *org);

    private:
        Experiment() {}
        void print(class Population *pop,
                   int experiment_num,
                   int generation);
        void evaluate(class Population *pop);

        const char *name;
        std::map<Test::Type, TestBattery> batteries;
        size_t ninputs;
        size_t noutputs;
    };
}
