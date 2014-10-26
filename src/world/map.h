#pragma once

namespace NEAT {

    struct Glyph {
        std::string type;
        char character;
        std::map<std::string, std::string> attrs;

        std::string str();
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

        bool try_find(std::string row, std::string col, Location &result);
    };

    struct Object {
        Location loc;
        Glyph glyph;
        std::map<std::string, std::string> attrs;
    };

    struct Map {
        LocationTranslator loc_trans;
        size_t width;
        size_t height;
        std::map<Location, Object> objects;
    };

    Map parse_map(std::string path);

}
