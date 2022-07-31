/*
 * utils.hpp
 */

#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <iostream>
#include <vector>
#include <filesystem>
#include <string>

#ifndef LOGLEVEL
#define LOGLEVEL 0
#endif

namespace Utils {
    enum loglevel {
        DEBUG,
        INFO,
        WARN,
        ERR,
    };

    void output_log(loglevel level, std::string str);
    std::vector<std::string> get_files(const std::string &path_str);
}

#endif