#include "neat.h"
#include "networkmanager.h"
#include "organism.h"
#include "util.h"

namespace NEAT {

    class Experiment {
    public:
        static Experiment *get(const char *name);
        static std::vector<std::string> get_names();

    private:
        static std::map<std::string, Experiment*> *experiments;

    public:
        virtual ~Experiment();

        void run(class rng_t &rng, int gens);

    protected:
        Experiment(const char *name);

        virtual void init_env() {}
        virtual void init_experiment() = 0;

        virtual BatchSensors *get_sensors() = 0;
        virtual void process_output(class Network &net, size_t istep) = 0;
        virtual OrganismEvaluation evaluate(Organism &org) = 0;
        virtual bool is_success(Organism *org);

        size_t ninputs;
        size_t noutputs;

    private:
        Experiment() {}
        void print(class OrganismEvaluator *eval,
                   int experiment_num,
                   int generation);
        void evaluate(class OrganismEvaluator *eval);

        const char *name;
    };

}
