#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "util.h"
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

void mkdir(const string &path) {
    int status = ::mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if(0 != status) {
        char buf[2048];
        sprintf(buf, "Failed making directory '%s'", path.c_str());
        perror(buf);
        exit(1);
    }
}

bool exists(const std::string &path) {
    struct stat buffer;
    return (stat (path.c_str(), &buffer) == 0);     
}

vector<string> permute_repeat(const string &letters,
                              size_t len) {
    vector<string> result;
    string buf;
    
    struct local {
        static void __permute(const string &letters,
                              size_t depth,
                              size_t len,
                              vector<string> &result,
                              string &buf) {
            if(depth == len) {
                result.push_back(buf);
            } else {
                for (size_t i = 0; i < letters.size(); ++i) {
                    buf.append(letters, i, 1);
                    __permute(letters, depth+1, len, result, buf);
                    buf.erase(buf.size() - 1);
                }
            }
        }
    };

    local::__permute(letters, 0, len, result, buf);
    
    return result;
}
