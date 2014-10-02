#include "neat.h"
#include "organism.h"
#include "util.h"

#include <map>
#include <vector>


namespace NEAT {

    struct Step {
        std::vector<real_t> input;
        std::vector<real_t> output;
        real_t weight;

        Step(const std::vector<real_t> &input_,
             const std::vector<real_t> &output_,
             real_t weight_ = 1.0);

        real_t err(class Network *net) const;
    };

    struct Test {
        enum Type {
            Training,
            Fittest
        };

        std::vector<Step> steps;
        bool is_training;
        Type type;

        Test(const std::vector<Step> &steps_, Type type_ = Training)
        : steps(steps_), type(type_) {
        }
    };

    struct TestBattery {
        std::vector<Test> tests;
        real_t max_err = 0.0;

        void add(const Test &test);

        struct EvalResult {
            real_t fitness;
            real_t error;
        };
        
        EvalResult evaluate(class Organism &org) const;
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
