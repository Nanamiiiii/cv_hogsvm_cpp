/*
 * utils.hpp
 */

#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <iostream>
#include <vector>
#include <filesystem>
#include <string>
#include <cstdlib>

namespace Utils {
    class Logger {
        public:
        Logger();
        void debug(std::string str);
        void info(std::string str);
        void warn(std::string str);
        void error(std::string str);

        private:
        int loglevel = 1;
    };
    std::vector<std::string> get_files(const std::string &path_str);
}

#endif