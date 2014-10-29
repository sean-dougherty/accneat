#include "std.hxx"

#include "mazeevaluator.h"
#include "map.h"
#include "network.h"
#include "networkexecutor.h"
#include "resource.h"
#include <assert.h>

//#define Truncate_Seq 2

#define Max_Seq_Len 3
#define Max_Maze_Len 26

using namespace std;

namespace NEAT {

    struct position_t {
        char row;
        char col;
        __net_eval_decl position_t() {
        }
        __net_eval_decl position_t(char row_, char col_) : row(row_), col(col_) {
        }
    };

    __net_eval_decl position_t operator+(const position_t a, const position_t b) {
        return position_t( char(a.row + b.row), char(a.col + b.col) );
    }

    __net_eval_decl bool operator==(const position_t a, const position_t b) {
        return (a.row == b.row) && (a.col == b.col);
    }

    enum direction_t {
        dir_east = 0, dir_north = 1, dir_west = 2, dir_south = 3
    };

    enum rotation_t {
        rot_clockwise = -1, rot_none = 0, rot_counter = 1
    };

    __net_eval_decl position_t get_rel_pos(const direction_t &dir) {
        switch(dir) {
        case dir_east: return position_t(0, 1);
        case dir_north: return position_t(-1, 0);
        case dir_west: return position_t(0, -1);
        case dir_south: return position_t(1, 0);
#ifdef ENABLE_CUDA
        default: return position_t(0, 0);
#else
        default: abort();
#endif
        }
    }

    __net_eval_decl direction_t rotate(const direction_t &dir, const rotation_t &rot) {
        return direction_t( (int(dir) + int(rot) + 4) % 4 );
    }

    __net_eval_decl position_t get_look_pos(const position_t &pos,
                            const direction_t &dir,
                            const rotation_t &rot) {
        return pos + get_rel_pos( rotate(dir,rot) );
    }

    ostream &operator<<(ostream &out, const direction_t &dir) {
        switch(dir) {
        case dir_north: return out << "north";
        case dir_south: return out << "south";
        case dir_west: return out << "west";
        case dir_east: return out << "east";
        default: abort();
        }
    }

    ostream &operator<<(ostream &out, const position_t &pos) {
        return out << "[" << (int)pos.row << "," << (int)pos.col << "]";
    }

    static direction_t parse(string &str) {
#define __if(name) if(str == #name) return dir_##name
        __if(east);
        __if(north);
        __if(west);
        __if(south);
        abort();
#undef __if
    }

    struct Config {
        uchar width;
        uchar height;
        position_t agent_pos;
        direction_t agent_dir;
        bool wall[Max_Maze_Len * Max_Maze_Len];
        struct Trial {
            position_t food_pos;
            uchar seqlen;
            real_t seq[Max_Seq_Len];
            ushort max_steps;
            uchar max_dist;
            uchar dist_map[Max_Maze_Len * Max_Maze_Len];
        };
        ushort ntrials;
        Trial trials[];
    };

    static void create_distance_map(const Config *config,
                                    position_t from,
                                    uchar *dist_map,
                                    uchar &max_dist) {
        const uchar w = config->width;
        const uchar h = config->height;

        memset(dist_map, 0xff, sizeof(uchar) * w * h);

        max_dist = 0;

        struct local {
            static void _(const uchar w,
                          const uchar h,
                          const bool *wall,
                          uchar *dist_map,
                          uchar &max_dist,
                          const uchar row, 
                          const uchar col,
                          const uchar dist) {
                if(row >= h) return;
                if(col >= w) return;

                size_t offset = w * row + col;

                if(dist >= dist_map[offset]) {
                    return;
                }
                if(wall[offset]) {
                    return;
                }

                dist_map[offset] = dist;
                if(dist > max_dist)
                    max_dist = dist;

                _(w, h, wall, dist_map, max_dist, row - 1, col, dist + 1);
                _(w, h, wall, dist_map, max_dist, row + 1, col, dist + 1);
                _(w, h, wall, dist_map, max_dist, row, col - 1, dist + 1);
                _(w, h, wall, dist_map, max_dist, row, col + 1, dist + 1);
            }
        };

        local::_(w, h, config->wall, dist_map, max_dist, from.row, from.col, 0);
    }

    static void create_config(__out Config *&config_,
                              __out size_t &len_) {
        Map map = parse_map( find_resource("maze.map") );

        Config config;
        config.width = map.width;
        config.height = map.height;
        assert(config.width * config.height <= sizeof(config.wall));

        config.ntrials = 0;

        memset(config.wall, 0, sizeof(bool) * config.width * config.height);

        vector<Config::Trial> trials;

        for(std::map<Location, Object>::iterator it = map.objects.begin();
            it != map.objects.end();
            it++) {

            Object &obj = it->second;
            size_t row = obj.loc.index.row;
            size_t col = obj.loc.index.col;

            if(obj.glyph.type == "wall") {
                config.wall[row * map.width + col] = true;
            }  else if(obj.glyph.type == "agent") {
                config.agent_pos.row = obj.loc.index.row;
                config.agent_pos.col = obj.loc.index.col;
                config.agent_dir = parse(obj.glyph.attrs["dir"]);
            } else if(obj.glyph.type == "food") {
                if( obj.attrs.find("seq") != obj.attrs.end() ) {
                    Config::Trial trial;
                    trial.food_pos.row = obj.loc.index.row;
                    trial.food_pos.col = obj.loc.index.col;
                    string seq = obj.attrs["seq"];
                    assert(seq.length() > 0 && seq.length() <= Max_Seq_Len);
            
                    trial.seqlen = seq.length();
                    for(size_t i = 0; i < seq.length(); i++) {
                        char c = seq[i];
                        if(c == 'l') {
                            trial.seq[i] = 0.0;
                        } else if(c == 'r') {
                            trial.seq[i] = 1.0;
                        } else {
                            abort();
                        }
                    }

#ifdef Truncate_Seq
                    for(size_t i = Truncate_Seq; i < seq.length(); i++) {
                        trial.seq[i] = 0.5;
                    }
#endif

                    trials.push_back(trial);
                }
            } else {
                abort();
            }
        }

        config.ntrials = trials.size();

        len_ = sizeof(Config) + sizeof(Config::Trial) * config.ntrials;
        config_ = (Config *)malloc(len_);
        memcpy(config_, &config, sizeof(Config));

        for(size_t i = 0; i < config.ntrials; i++) {
            Config::Trial &trial = trials[i];
            create_distance_map(config_, trial.food_pos, trial.dist_map, trial.max_dist);
            uchar dist = trial.dist_map[config.agent_pos.row * config.width + config.agent_pos.col];
            trial.max_steps = dist + 3 * trial.seqlen;
        }
        memcpy(config_->trials, trials.data(), sizeof(Config::Trial) * config.ntrials);

/*
        for(size_t i = 0; i < config.ntrials; i++) {
            printf("max_steps=%u\n", unsigned(config_->trials[i].max_steps));
            printf("max_dist=%u\n", unsigned(config_->trials[i].max_dist));
            uchar *dist_map = config_->trials[i].dist_map;
            const uchar w = config_->width;
            const uchar h = config_->height;
            for(uchar row = 0; row < h; row++) {
                for(uchar col = 0; col < w; col++) {
                    uchar dist = dist_map[row * w + col];
                    if(dist == uchar(0xff))
                        printf("..");
                    else
                        printf("%02d", int(dist));
                }
                printf("\n");
            }
        }

        exit(0);
*/
/*
  for(size_t row = 0; row < map.height; row++) {
  for(size_t col = 0; col < map.width; col++) {
  if(config.wall[row * map.width + col]) {
  cout << "*";
  } else {
  if( (row == config.agent_row) && (col == config.agent_col) ) {
  cout << "A";
  } else {
  bool trial = false;
  size_t i;
  for(i = 0; i < trials.size(); i++) {
  if(row == trials[i].food_row && col == trials[i].food_col) {
  trial = true;
  break;
  }
  }
  if(trial) {
  cout << "f";
  } else {
  cout << ".";
  }
  }
  }
  }
  cout << endl;
  }
*/
    }

    struct Evaluator {
        typedef NEAT::Config Config;

        const Config *config;
        ushort trial;
        ushort trial_step;
        position_t agent_pos;
        direction_t agent_dir;
        position_t food_pos;
        ushort max_trial_steps;
        const uchar *dist_map;
        uchar best_dist;
        char iseq;
        OrganismEvaluation eval;

        __net_eval_decl Evaluator(const Config *config_)
        : config(config_) {

            trial = 0;
            trial_step = 0;
            max_trial_steps = trial_step + 1;
            food_pos = position_t(0,0);
            agent_pos = position_t(0,0);
            agent_dir = dir_east;
            dist_map = NULL;
            best_dist = uchar(0xff);
            iseq = 0;
            eval.error = 0.0;
            eval.fitness = 0.0;
        }

        __net_eval_decl uchar dist() {
            return dist_map[config->width * agent_pos.row + agent_pos.col];
        }

        __net_eval_decl bool next_step() {
            bool trial_complete = (best_dist == 0) || (trial_step == max_trial_steps);
            if(trial_complete) {
                if(trial == config->ntrials - 1) {
                    return false;
                } else {
                    trial++;
                    trial_step = 1;
                }
            } else {
                trial_step++;
            }

            if(trial_step == 1) {
                agent_pos = config->agent_pos;
                agent_dir = config->agent_dir;
                food_pos = config->trials[trial].food_pos;
                max_trial_steps = config->trials[trial].max_steps;
                dist_map = config->trials[trial].dist_map;
                best_dist = dist();
                iseq = 0;
            } else {
                if(trial_step <= config->trials[trial].seqlen * 2) {
                    if(trial_step % 2 == 0) {
                        iseq = -1;
                    } else {
                        iseq = trial_step / 2;
                    }
                } else {
                    iseq = -2;
                }
            }

            return true;
        }

        __net_eval_decl bool clear_noninput() {
            return trial_step == 1;
        }

        __net_eval_decl real_t obj_sensor(rotation_t rot) {
            position_t look_pos = get_look_pos(agent_pos, agent_dir, rot);
            return config->wall[int(look_pos.row) * config->width + int(look_pos.col)]
                ? 1.0
                : 0.0;
        }

        __net_eval_decl real_t get_sensor(node_size_t sensor_index) {
            switch( sensor_t(sensor_index) ) {
            case sensor_right:
                return obj_sensor(rot_clockwise);
            case sensor_fwd:
                return obj_sensor(rot_none);
            case sensor_left:
                return obj_sensor(rot_counter);
            case sensor_sound:
                return iseq > -1 ? 1.0 : 0.0;
            case sensor_freq:
                return iseq > -1 ? config->trials[trial].seq[int(iseq)] : 0.0;
            case sensor_go:
                return iseq == -2 ? 1.0 : 0.0;
            default:
#ifdef ENABLE_CUDA
                return 0.0;
#else
                abort();
#endif
            }
        }

        __net_eval_decl void evaluate(real_t *output) {
            if(iseq == -2) {
                if(output[output_right] > 0.5) {
                    agent_dir = rotate(agent_dir, rot_clockwise);
                }
                if(output[output_left] > 0.5) {
                    agent_dir = rotate(agent_dir, rot_counter);
                }
                if(output[output_fwd] > 0.5) {
                    position_t newpos = agent_pos + get_rel_pos(agent_dir);
                    if(!config->wall[int(newpos.row) * config->width + int(newpos.col)]) {
                        agent_pos = newpos;
                    }
                }
                uchar curr_dist = dist();
                if(curr_dist < best_dist)
                    best_dist = curr_dist;

                if(best_dist == 0) {
                    eval.fitness += 10.0 + (max_trial_steps - trial_step) / real_t(max_trial_steps);
                } else if(trial_step == max_trial_steps) {
                    eval.error += 1.0;
                    uchar max_dist = config->trials[trial].max_dist;
                    eval.fitness += real_t(max_dist - best_dist) / real_t(max_dist);
                    eval.fitness += real_t(max_dist - curr_dist) / real_t(max_dist);
                }
            }
        }

        __net_eval_decl OrganismEvaluation result() {
            return eval;
        }

    };

    class MazeEvaluator : public NetworkEvaluator {
        NetworkExecutor<Evaluator> *executor;
    public:
        MazeEvaluator() {
            executor = NetworkExecutor<Evaluator>::create();
            Evaluator::Config *config;
            size_t configlen;
            create_config(config, configlen);
            executor->configure(config, configlen);
            free(config);
        }

        ~MazeEvaluator() {
            delete executor;
        }

        virtual void execute(class Network **nets_,
                             class OrganismEvaluation *results,
                             size_t nnets) {
            executor->execute(nets_, results, nnets);
        }
    };

    class NetworkEvaluator *create_maze_evaluator() {
        return new MazeEvaluator();
    }

}
