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

        real_t err(class Network *net,
                   float **details_act,
                   float **details_err);
    };

    struct Test {
        std::vector<Step> steps;
    };

    class Experiment {
    public:
        static Experiment *get(const char *name);

    private:
        static std::map<std::string, Experiment*> *experiments;

    public:
        virtual ~Experiment();

        void init();
        void run(class rng_t &rng, int gens);

    protected:
        Experiment(const char *name);
        virtual void init_env() {}
        virtual std::vector<Test> create_tests() = 0;
        virtual bool is_success(class Organism *org);

    private:
        Experiment() {}
        real_t score(real_t errorsum);
        void print(class Population *pop,
                   int experiment_num,
                   int generation);
        void evaluate(class Population *pop);
        void evaluate_org(class Organism &org);

        const char *name;
        std::vector<Test> tests;
        size_t nsteps;
        size_t nouts;
        real_t max_err;
        float *details_act = nullptr;
        float *details_err = nullptr;
    };
}
