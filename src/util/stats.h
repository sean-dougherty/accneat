#pragma once

namespace NEAT {
    template<typename T, typename U=T>
    U sum(const std::vector<T> &vec) {
        U result = 0;
        for(const T &x: vec) result += (U)x;
        return result;
    }

    template<typename T, typename U=double>
    U mean(const std::vector<T> &vec) {
        if(vec.size() == 0) return NAN;
        return sum<T,U>(vec) / vec.size();
    }

    template<typename T>
    T min(const std::vector<T> &vec) {
        T minval = vec.front();
        for(size_t i = 1; i < vec.size(); i++) {
            if(vec[i] < minval)
                minval = vec[i];
        }
        return minval;
    }

    template<typename T>
    T max(const std::vector<T> &vec) {
        T maxval = vec.front();
        for(size_t i = 1; i < vec.size(); i++) {
            if(vec[i] > maxval)
                maxval = vec[i];
        }
        return maxval;
    }

    struct stats_t {
        size_t n;
        double min;
        double max;
        double mean;
    };

    inline std::ostream &operator<<(std::ostream &out, const stats_t &stats) {
        return out << "n=" << stats.n
                   << ", min=" << stats.min
                   << ", max=" << stats.max
                   << ", mean=" << stats.mean;
    }

    template<typename T>
    stats_t stats(const std::vector<T> &vec) {
        stats_t result;

        result.n = vec.size();
        if(result.n > 0) {
            result.min = min(vec);
            result.max = max(vec);
            result.mean = mean<T,double>(vec);
        } else {
            result.min = result.max = result.mean = 0;
        }

        return result;
    }
}
