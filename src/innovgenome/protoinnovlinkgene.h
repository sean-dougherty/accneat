#pragma once

namespace NEAT {

    class ProtoInnovLinkGene {
        InnovGenome *_genome = nullptr;
        //todo: does this have to be a InnovLinkGene* now?
        InnovLinkGene *_gene = nullptr;
        InnovNodeGene *_in = nullptr;
        InnovNodeGene *_out = nullptr;
    public:
        void set_gene(InnovGenome *genome, InnovLinkGene *gene) {
            _genome = genome;
            _gene = gene;
        }
        InnovLinkGene *gene() {
            return _gene;
        }

        void set_out(InnovNodeGene *out) {
            _out = out;
            _gene->set_out_node_id(out->node_id);
        }
        InnovNodeGene *out() {
            return _out ? _out : _genome->get_node(_gene->out_node_id());
        }

        void set_in(InnovNodeGene *in) {
            _in = in;
            _gene->set_in_node_id(in->node_id);
        }
        InnovNodeGene *in() {
            return _in ? _in : _genome->get_node(_gene->in_node_id());
        }
    };

}
