#pragma once

namespace NEAT {

    //---
    //--- CLASS CpuNetwork
    //---
	class CpuNetwork : public Network {
    private:
        NetDims dims;
		std::vector<NetNode> nodes;
		std::vector<NetLink> links;
        std::vector<real_t> activations;

    public:
        CpuNetwork() {}
		virtual ~CpuNetwork() {}

		void activate(size_t ncycles);
        std::vector<real_t> &get_activations(__out std::vector<real_t> &result);
        void set_activations(__in std::vector<real_t> &newacts);

        void clear_noninput();
        void load_sensor(size_t isensor, real_t activation);
        real_t *get_outputs();

        virtual void configure(const NetDims &dims,
                               NetNode *nodes,
                               NetLink *links) override;

        virtual NetDims get_dims() override { return dims; }

	};

    //---
    //--- CLASS CpuNetworkExecutor
    //---
    template<typename Evaluator>
    class CpuNetworkExecutor : public NetworkExecutor<Evaluator> {
    public:
        const typename Evaluator::Config *config = nullptr;

        virtual ~CpuNetworkExecutor() {
            delete config;
        }

        virtual void configure(const typename Evaluator::Config *config_,
                               size_t len) override {
            void *buf = malloc(len);
            memcpy(buf, config_, len);
            config = (const typename Evaluator::Config *)buf;
        }

        virtual void execute(class Network **nets_,
                             OrganismEvaluation *results,
                             size_t nnets) override {

            CpuNetwork **nets = (CpuNetwork **)nets_;
            size_t nsensors = nets[0]->get_dims().nnodes.sensor;

#pragma omp parallel for
            for(size_t inet = 0; inet < nnets; inet++) {
                CpuNetwork *net = nets[inet];
                Evaluator eval{config};

                for(size_t istep = 0; !eval.complete(istep); istep++) {
                    if(eval.clear_noninput(istep)) {
                        net->clear_noninput();
                    }
                    for(size_t isensor = 0; isensor < nsensors; isensor++) {
                        net->load_sensor(isensor, eval.get_sensor(istep, isensor));
                    }
                    net->activate(NACTIVATES_PER_INPUT);
                    eval.evaluate(istep, net->get_outputs());
                }

                results[inet] = eval.result();
            }
        }
        
    };

    inline std::unique_ptr<class Network> create_default_network() {
        return std::unique_ptr<Network>(new CpuNetwork());
    }

    template<typename Evaluator>
    inline NetworkExecutor<Evaluator> *create_network_executor() {
        return new CpuNetworkExecutor<Evaluator>();
    }

}
