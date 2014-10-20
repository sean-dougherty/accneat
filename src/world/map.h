#pragma once

#include "util.h"

struct Glyph {
    std::string type;
    char character;
    std::map<std::string, std::string> attrs;

    std::string str() {
        std::ostringstream out;
        out << "Glyph{"
            << "type=" << type
            << ", char=" << character;

        out << ", attrs={";
        bool first = true;
        for(auto &kv: attrs) {
            if(!first) {
                out << ", ";
                first = false;
            }
            out << kv.first << "='" << kv.second << "'";
        }
        out << "}";
        out << "}";
        return out.str();
    }
};

struct Location {
    struct {
        std::string row;
        std::string col;
    } label;
    struct {
        size_t row;
        size_t col;
    } index;

    bool operator<(const Location &other) const {
        if(index.row < other.index.row) {
            return true;
        } else if(index.row == other.index.row) {
            return index.col < other.index.col;
        } else {
            return false;
        }
    }
};

struct LocationTranslator {
    std::map<std::string, size_t> row_index;
    std::map<std::string, size_t> col_index;
    std::map<size_t, std::string> col_label;
    std::map<size_t, std::string> row_label;

    bool try_find(std::string row, std::string col, Location &result) {
        if(::try_find(row_index, row, result.index.row)) {
            if(::try_find(col_index, col, result.index.col)) {
                result.label.row = row;
                result.label.col = col;
                return true;
            }
        }

        result = {};
        return false;
    }
};

struct Object {
    Location loc;
    Glyph glyph;
    std::map<std::string, std::string> attrs;
};

struct Map {
    LocationTranslator loc_trans;
    std::map<Location, Object> objects;
};

Map parse_map(std::string path);
