/*
 * main.cpp
 */

#include <iostream>
#include <vector>
#include <filesystem>
#include <string>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"

#define LOGLEVEL 0

namespace Utils {
    enum loglevel {
        DEBUG,
        INFO,
        WARN,
        ERR,
    };

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

    std::vector<std::string> get_files(std::string path_str) {
        using namespace std::filesystem;
        std::vector<std::string> files;
        path target_dir(path_str);

        if (!is_directory(target_dir)) {
            output_log(ERR, "specified path is not directory");
            return files;
        }

        for (const auto &file : directory_iterator(path_str)) {
            std::string filename = file.path().string();
            output_log(DEBUG, "found: " + filename);
            files.push_back(filename);
        }
        return files;
    }
}

int main(void) {
    return 0;
}
