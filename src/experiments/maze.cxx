#include "std.hxx"

#include "map.h"
#include "network.h"
#include "networkexecutor.h"
#include <assert.h>

using namespace NEAT;
using namespace std;

enum direction_t {
    dir_right = 0, dir_up = 1, dir_left = 2, dir_down = 3
};

static direction_t parse(string &str) {
#define __if(name) if(str == #name) return dir_##name

    __if(right);
    __if(up);
    __if(left);
    __if(down);
    abort();
#undef __if
}

enum input_t {
    obj_right = 0, obj_fwd = 1, obj_left = 2
};

enum output_t {
    turn_right = 0, turn_left = 1, move_fwd = 2
};

#define Max_Seq_Len 3
#define Max_Trial_Steps 50

struct Config {
    uchar width;
    uchar height;
    uchar agent_row;
    uchar agent_col;
    direction_t agent_dir;
    bool wall[32*32];
    struct Trial {
        uchar food_row;
        uchar food_col;
        uchar seqlen;
        real_t seq[Max_Seq_Len];
    };
    ushort ntrials;
    Trial trials[];
};

static void create_config(__out Config *&config_,
                          __out size_t &len_) {
    Map map = parse_map("./res/maze.map");

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
            config.agent_row = obj.loc.index.row;
            config.agent_col = obj.loc.index.col;
            config.agent_dir = parse(obj.glyph.attrs["dir"]);
        } else if(obj.glyph.type == "food") {
            Config::Trial trial;
            trial.food_row = obj.loc.index.row;
            trial.food_col = obj.loc.index.col;
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

            trials.push_back(trial);
        } else {
            abort();
        }
    }

    config.ntrials = trials.size();

    len_ = sizeof(Config) + sizeof(Config::Trial) * config.ntrials;
    config_ = (Config *)malloc(len_);
    memcpy(config_, &config, sizeof(Config));
    memcpy(config_->trials, trials.data(), sizeof(Config::Trial) * config.ntrials);

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

/*
static struct MazeInit {
    MazeInit() {
        Config *config; size_t len;
        create_config(config, len);
    }
} init;


struct Evaluator {
    typedef ::Config Config;

    const Config *config;
    ushort trial;
    ushort trial_step;
    bool success;

    Evaluator(const Config *config_)
    : config(config_) {

        trial = 0;
        trial_step = 0;
        success = false;
    }

    __net_eval_decl bool next(size_t _) {
        bool trial_complete = success || (trial_step == Max_Trial_Steps);
        if(trial_complete) {
            if(trial == config->ntrials - 1) {
                return false;
            } else {
                trial++;
                trial_step = 1;
                success = false;
            }
        } else {
            trial_step++;
        }
    }

    __net_eval_decl bool clear_noninput(size_t _) {
        return trial_step == 1;
    }

    __net_eval_decl real_t get_sensor(size_t _,
                                      size_t sensor_index) {
        return config->inputs(istep)[sensor_index];
    }

    __net_eval_decl void evaluate(size_t istep, real_t *actual) {
        real_t *expected = config->outputs(istep);
        real_t result = 0.0;

        for(size_t i = 0; i < config->noutputs; i++) {
            real_t err = actual[i] - expected[i];
            if(err < 0) err *= -1;
            if(err < 0.05) {
                err = 0.0;
            }
            result += err;
        }

        errorsum += result * config->parms(istep)->weight;
    }

    __net_eval_decl OrganismEvaluation result() {
        OrganismEvaluation eval;
        eval.error = errorsum;
        eval.fitness = 1.0 - errorsum/config->max_err;
        return eval;
    }

};

class MazeEvaluator : public NetworkEvaluator {
    NetworkExecutor<Evaluator> *executor;
public:
    MazeEvaluator() {
        executor = NetworkExecutor<Evaluator>::create();
    }

    ~MazeEvaluator() {
    }
}
*/
