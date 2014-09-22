#pragma once

namespace NEAT {

    class RecurrencyChecker {
    private:
        size_t nnodes;
        InnovLinkGene **links;
        size_t nlinks;

        static bool cmp_sort(const InnovLinkGene *x, const InnovLinkGene *y) {
            return x->out_node_id() < y->out_node_id();
        }

        static bool cmp_find(const InnovLinkGene *x, int node_id) {
            return x->out_node_id() < node_id;
        }

        bool find(int node_id, InnovLinkGene ***curr) {
            if(*curr == nullptr) {
                auto it = std::lower_bound(links, links + nlinks, node_id, cmp_find);
                if(it == links + nlinks) return false;
                if((*it)->out_node_id() != node_id) return false;

                *curr = it;
                return true;
            } else {
                (*curr)++;
                if(*curr >= (links + nlinks)) return false;
                if((**curr)->out_node_id() != node_id) return false;
                return true;
            }
        }

        // This checks a POTENTIAL link between a potential in_node and potential out_node to see if it must be recurrent 
        bool is_recur(int in_id, int out_id, int &count, int thresh) {
            ++count;  //Count the node as visited
            if(count > thresh) {
                return false;  //Short out the whole thing- loop detected
            }

            if (in_id==out_id) return true;
            else {
                InnovLinkGene **gene = nullptr;
                while( find(in_id, &gene) ) {
                    //But skip links that are already recurrent
                    //(We want to check back through the forward flow of signals only
                    if(!(*gene)->is_recurrent()) {
                        if( is_recur((*gene)->in_node_id(), out_id, count, thresh) )
                            return true;
                    }
                }
                return false;
            }
        }

    public:
        RecurrencyChecker(size_t nnodes_,
                          std::vector<InnovLinkGene> &genome_links,
                          InnovLinkGene **buf_links) {
            nnodes = nnodes_;
            links = buf_links;

            nlinks = 0;
            for(size_t i = 0; i < genome_links.size(); i++) {
                InnovLinkGene *g = &genome_links[i];
                if(g->enable) {
                    links[nlinks++] = g;
                }
            }
            std::sort(links, links + nlinks, cmp_sort);
        }

        bool is_recur(int in_node_id, int out_node_id) {
            //These are used to avoid getting stuck in an infinite loop checking
            //for recursion
            //Note that we check for recursion to control the frequency of
            //adding recurrent links rather than to prevent any paricular
            //kind of error
            int thresh=nnodes*nnodes;
            int count = 0;

            if(is_recur(in_node_id, out_node_id, count, thresh)) {
                return true;
            }

            //ADDED: CONSIDER connections out of outputs recurrent
            //todo: this was fixed to use place instead of type,
            //      but not clear if this logic is desirable. Shouldn't it
            //      just be checking if the output node is OUTPUT?
            /*
              if (((in_node->place)==OUTPUT)||
              ((out_node->place)==OUTPUT))
              return true;
            */
            return false;
        }
    
    };

}
