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
