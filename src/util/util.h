#pragma once

namespace NEAT {

#define error(msg) {std::cerr << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl; exit(1);}
#define warn(msg) {std::cerr << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl;}
#define trap(msg) {std::cerr << __FILE__ << ":" << __LINE__ << ": " << msg << std::endl; abort();}
#define impl() {std::cerr << __FILE__ << ":" << __LINE__ << ": IMPLEMENT!" << std::endl; abort();}
#define panic() {std::cerr << __FILE__ << ":" << __LINE__ << ": PANIC!" << std::endl; abort();}

#define sh(cmd) {int rc = system(cmd); if(rc != 0) error("Failed executing " << cmd);}

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

template<typename T, typename U>
bool contains(const T &container, const U &value) {
    return container.find(value) != container.end();
}

template<typename T, typename U>
void fill(T &container, const U &value) {
    std::fill(container.begin(), container.end(), value);
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

template<typename T, typename U>
bool try_find(const std::map<T, U> &m, const T &key, U &result) {
    auto it = m.find(key);
    if(it == m.end())
        return false;
    result = it->second;
    return true;
}

template<typename T, typename U>
bool try_find(std::map<T, U> &m, const T &key, U **result) {
    auto it = m.find(key);
    if(it == m.end())
        return false;
    *result = &it->second;
    return true;
}

inline std::string operator+(const char *c_str, const std::string &str) {
    return std::string(c_str) + str;
}

inline std::string str(char c) {
    return std::string(1, c);
}

#define WHITESPACE " \f\n\r\t\v"

inline std::string trim_right(
    const std::string& s,
    const std::string& delimiters = WHITESPACE )
{
    size_t pos = s.find_last_not_of( delimiters );
    return pos == std::string::npos ? "" : s.substr( 0, pos + 1 );
}

inline std::string trim_left(
    const std::string& s,
    const std::string& delimiters = WHITESPACE )
{
    size_t pos = s.find_first_not_of( delimiters );
    return pos == std::string::npos ? "" : s.substr( pos );
}

inline std::string trim(
    const std::string& s,
    const std::string& delimiters = WHITESPACE )
{
    return trim_left( trim_right( s, delimiters ), delimiters );
}

inline std::vector<std::string> split(const std::string& s,
                                      const std::string& delims = WHITESPACE,
                                      bool keep_empty = false) {
    if (delims.empty()) {
        return {s};
    }

    std::vector<std::string> result;
    size_t substart = 0;
    while (substart < s.length()) {
        size_t subend = s.find_first_of(delims, substart);
        if(subend == std::string::npos) {
            subend = s.length();
        }
        if( (subend != substart) || keep_empty) {
            result.push_back( s.substr(substart, subend - substart) );
        }
        substart = subend + 1;
    }
    return result;
}

void mkdir(const std::string &path);
bool exists(const std::string &path);

// e.g. ("ab", 3) --> {"aaa", "aab", "aba", "abb", "baa", "bab", "bba", "bbb"}
std::vector<std::string> permute_repeat(const std::string &letters, size_t len);
