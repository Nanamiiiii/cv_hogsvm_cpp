/*
 * utils.cpp
 */

#include "utils.hpp"

namespace Utils {
    void output_log(loglevel level, std::string str) {
        using namespace std;
        switch (level) {
            case INFO:
            #if LOGLEVEL <= 1
                cout << "[INFO]\t" << str << endl;
            #endif
                break;
            case WARN:
            #if LOGLEVEL <= 2
                cout << "[WARN]\t" << str << endl;
            #endif
                break;
            case ERR:
            #if LOGLEVEL <= 3
                cout << "[ERROR]\t" << str << endl;
            #endif
                break;
            case DEBUG:
            #if LOGLEVEL == 0
                cout << "[DEBUG]\t" << str << endl;
            #endif
                break;
        }
    }

    std::vector<std::string> get_files(const std::string &path_str) {
        using namespace std::filesystem;
        std::vector<std::string> files;
        path target_dir(path_str);

        Utils::output_log(Utils::DEBUG, "search directory: " + path_str);

        if (!is_directory(target_dir)) {
            output_log(ERR, "specified path is not directory: " + path_str);
            return files;
        }

        for (const auto &file : directory_iterator(path_str)) {
            std::string filename = file.path().string();
            output_log(DEBUG, "found: " + filename);
            files.push_back(filename);
        }
        output_log(INFO, std::to_string(files.size()) + " file(s) found in " + path_str);
        return files;
    }

}