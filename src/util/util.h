#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace NEAT {

#define error(msg) {std::cerr << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl; exit(1);}
#define warn(msg) {std::cerr << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl;}
#define trap(msg) {std::cerr << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl; abort();}
#define impl() {std::cerr << __FILE__ << ":" << __LINE__ << ": IMPLEMENT!" << std::endl; abort();}
#define panic() {std::cerr << __FILE__ << ":" << __LINE__ << ": PANIC!" << std::endl; abort();}

    template<typename Container, typename Predicate>
    void erase_if(Container &cont, Predicate predicate) {
        auto iterator = std::remove_if(cont.begin(), cont.end(), predicate);
        cont.erase(iterator, cont.end());
    }

    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}

template<typename T>
void append(std::vector<T> &a, const std::vector<T> &b) {
    a.insert(a.end(), b.begin(), b.end());
}

template<typename T>
void append(std::vector<T> &vec, const T &val, size_t n = 1) {
    for(size_t i = 0; i < n; i++) {
        vec.push_back(val);
    }
}

template<typename T>
std::vector<T> concat(const std::vector<T> &a, const std::vector<T> &b) {
    std::vector<T> c;
    append(c, a);
    append(c, b);
    return c;
}

void mkdir(const std::string &path);

// e.g. ("ab", 3) --> {"aaa", "aab", "aba", "abb", "baa", "bab", "bba", "bbb"}
std::vector<std::string> permute_repeat(const std::string &letters, size_t len);
