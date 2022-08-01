/*
 * utils.cpp
 */

#include "utils.hpp"

namespace Utils {
    Logger::Logger() {
        char *log_env = std::getenv("LOGLEVEL");
        this->loglevel = log_env ? atoi(log_env) : 1;
    }

    void Logger::debug(std::string str) {
        if (this->loglevel == 0)
            std::cout << "[DEBUG]\t" << str << std::endl;
    }

    void Logger::info(std::string str) {
        if (this->loglevel >= 1)
            std::cout << "[INFO]\t" << str << std::endl;
    }

    void Logger::warn(std::string str) {
        if (this->loglevel >= 2)
            std::cout << "[WARN]\t" << str << std::endl;
    }

    void Logger::error(std::string str) {
        if (this->loglevel == 3)
            std::cout << "[ERROR]\t" << str << std::endl;
    }

    std::vector<std::string> get_files(const std::string &path_str) {
        using namespace std::filesystem;
        std::vector<std::string> files;
        path target_dir(path_str);
        Logger logger = Logger();

        logger.debug("search directory: " + path_str);

        if (!is_directory(target_dir)) {
            logger.error("specified path is not directory: " + path_str);
            return files;
        }

        for (const auto &file : directory_iterator(path_str)) {
            std::string filename = file.path().string();
            logger.debug("found: " + filename);
            files.push_back(filename);
        }
        logger.info(std::to_string(files.size()) + " file(s) found in " + path_str);
        return files;
    }

}