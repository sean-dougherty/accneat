#include "std.h" // Must be included first. Precompiled header with standard library includes.
#include "map.h"
#include "util.h"

using namespace std;

namespace NEAT {
    static const set<string> Valid_Sections = {
        "glyphs",
        "glyphs-attrs",
        "objects",
        "objects-attrs"
    };

    string Glyph::str() {
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

    bool LocationTranslator::try_find(std::string row, std::string col, Location &result) {
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

    struct Line {
        string path;
        size_t lineno;
        string text;

        void err(string msg) {
            cerr << path << ":" << lineno << ": " << msg << endl;
            exit(1);
        }
    };

    struct Section {
        string name;
        vector<Line> lines;
    };

    static map<string, Section> parse_sections(string path) {
        ifstream in(path);
        if(!in.good()) {
            error("Failed opening " << path);
        }

        map<string, Section> result;

        size_t lineno = 0;
        auto err = [&lineno, &path] (const string &msg) {
            cerr << path << ":" << lineno << ": " << msg << endl;
            exit(1);
        };

        Section section;

        while(!in.eof()) {
            char buf[1024 * 32];
            in.getline(buf, sizeof(buf));
            if(in.bad()) {
                error("Failed reading from " << path);
            }
        
            lineno++;

            string text{buf};
            {
                size_t pos = text.find_first_of('#');
                if(pos != string::npos)
                    text.resize(pos);
            }
            text = trim(text);

            if(text.empty()) {
                continue;
            }

            if(text.front() == '[') {
                if(text.back() != ']') {
                    err("Expecting ']' as final character");
                }

                if(!section.name.empty()) {
                    result[section.name] = section;
                    section = {};
                }

                section.name = trim( text.substr(1, text.length() - 2) );
                if(section.name.empty()) {
                    err("Section name cannot be empty");
                } else if(Valid_Sections.find(section.name) == Valid_Sections.end()) {
                    err("Invalid section name");
                }
            } else {
                if(section.name.empty()) {
                    err("No section header");
                }

                section.lines.push_back({path, lineno, text});
            }
        }

        if(section.name.empty()) {
            err("Expected to find at least one section.");
        }
        result[section.name] = section;

        for(auto name: Valid_Sections) {
            if(result.find(name) == result.end()) {
                err("Missing section " + name);
            }
        }

        return result;
    }

    static pair<string, string> parse_equals(Line &line) {
        size_t pos = line.text.find_first_of('=');
        if(pos == string::npos) {
            line.err("Expecting '='");
        }
        string lhs = trim( line.text.substr(0, pos) );
        string rhs = trim( line.text.substr(pos + 1) );

        return {lhs, rhs};
    }

    static void parse_glyphs_attrs(Section &section, map<char, Glyph> &glyphs) {
        for(Line &line: section.lines) {
            pair<string, string> def = parse_equals(line);

            auto syn_err = [&line]() {
                line.err("Expecting: char name = value (e.g. '> dir = right')");
            };

            vector<string> lhs_tokens = split(def.first);
            if(lhs_tokens.size() != 2) {
                syn_err();
            }
            if(lhs_tokens.front().length() != 1) {
                syn_err();
            }

            char c = lhs_tokens.front()[0];
            if(glyphs.find(c) == glyphs.end()) {
                line.err("Undefined glyph: '" + str(c) + "'");
            }
            string name = lhs_tokens.back();
            Glyph &g = glyphs[c];
            if(g.attrs.find(name) != g.attrs.end()) {
                line.err("Duplicate definition");
            }
            string value = def.second;
            g.attrs[name] = value;
        }
    }

    static map<char, Glyph> parse_glyphs(Section &section_glyphs,
                                         Section &section_attrs) {
        map<char, Glyph> glyphs;

        for(Line &line: section_glyphs.lines) {
            pair<string, string> def = parse_equals(line);

            string type = def.first;
            for(char c: def.second) {
                if(isspace(c)) {
                    continue;
                }
                if(glyphs.find(c) != glyphs.end()) {
                    line.err("duplicate glyph: " + str(c));
                }

                glyphs[c] = {type, c};
            }
        }

        parse_glyphs_attrs(section_attrs, glyphs);

        return glyphs;
    }

    static void parse_objects_attrs(Section &section,
                                    Map &map) {
        for(Line &line: section.lines) {
            pair<string, string> def = parse_equals(line);

            auto syn_err = [&line](string details) {
                line.err(details+".\nExpecting: row,col glyph name = value (e.g. 'A,C f seq = llr')");
            };

            vector<string> lhs_tokens = split(def.first);
            if(lhs_tokens.size() != 3) {
                syn_err("Too many tokens on lhs");
            }
            vector<string> coord = split(lhs_tokens.front(), ",");
            if(coord.size() != 2) {
                syn_err("Coordinate is malformed: '"+lhs_tokens.front()+"'");
            }

            if(lhs_tokens[1].length() != 1) {
                syn_err("Glyph should be only 1 character, found ("+lhs_tokens[1]+")");
            }

            Location loc;
            if(!map.loc_trans.try_find(coord[0], coord[1], loc)) {
                line.err("Invalid coordinate");
            }
            Object *obj;
            if(!try_find(map.objects, loc, &obj)) {
                line.err("No object at coordinate.");
            }

            char c = lhs_tokens[1][0];
            if(c != obj->glyph.character) {
                line.err("Specified glyph doesn't match that found at coordinate (" + str(c) + ")");
            }

            if( contains(obj->attrs, lhs_tokens[2]) ) {
                line.err("Duplicate attr definition");
            }
            obj->attrs[lhs_tokens[2]] = def.second;
        }    
    }

    static Map parse_objects(Section &section,
                             Section &section_attrs,
                             map<char, Glyph> &glyphs) {
        Map map;

        {
            Line &line = section.lines.front();
            string header = line.text;
            if(header[0] != '+') {
                line.err("Column labels line must start with '+'");
            }
            string labels = header.substr(1);
            if( labels.find_first_of(WHITESPACE) != string::npos ) {
                line.err("Unexpected whitespace");
            }
            if( labels.find_first_of(',') != string::npos ) {
                line.err("Unexpected comma");
            }
            map.width = labels.length();

            for(char c: labels) {
                string label = str(c);
                if(contains(map.loc_trans.col_index, label)) {
                    line.err("Duplicate column label '" + label + "'");
                }

                size_t index = map.loc_trans.col_index.size();
                map.loc_trans.col_index[label] = index;
                map.loc_trans.col_label[index] = label;
            }
        }

        map.height = 0;

        for(size_t i = 1; i < section.lines.size(); i++) {
            Line &line = section.lines[i];

            string row_label = str(line.text[0]);
            string content = line.text.substr(1);
            if(content.length() > map.loc_trans.col_index.size()) {
                line.err("Columns exceeded");
            }

            size_t row = i - 1;
            map.height++;

            map.loc_trans.row_index[row_label] = row;
            map.loc_trans.row_label[row] = row_label;

            for(size_t j = 0; j < content.size(); j++) {
                char c = content[j];

                if(c == '\t') {
                    line.err("Tabs not allowed");
                } else if(c != ' ') {
                    if(!contains(glyphs, c)) {
                        line.err("Undefined glyph: '" + str(c) + "'");
                    }

                    size_t col = j;

                    Location location{{row_label, map.loc_trans.col_label[col]}, {row, col}};
                    Glyph &glyph = glyphs[c];

                    map.objects[location] = {location, glyph};
                }
            }
        }

        parse_objects_attrs(section_attrs, map);

        return map;
    }

    Map parse_map(string path) {
        map<string, Section> sections = parse_sections(path);

        map<char, Glyph> glyphs = parse_glyphs(sections["glyphs"],
                                               sections["glyphs-attrs"]);

        Map map = parse_objects(sections["objects"],
                                sections["objects-attrs"],
                                glyphs);

        return map;
    }

}
