#pragma once

namespace NEAT {

    class Experiment {
    public:
        static Experiment *get(const char *name);
        static std::vector<std::string> get_names();

    private:
        static std::map<std::string, Experiment*> *experiments;

    public:
        virtual ~Experiment();

        virtual void run(class rng_t &rng, int gens) = 0;

    protected:
        Experiment(const char *name);

    private:
        Experiment() {}

        const char *name;
    };
}
