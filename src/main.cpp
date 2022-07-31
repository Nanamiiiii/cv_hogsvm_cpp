/*
 * main.cpp
 */

#include "utils.hpp"
#include "hogsvm.hpp"
#include <boost/program_options.hpp>

int main(const int argc, const char* const * const argv) {
    using namespace boost::program_options;

    /* Definition of command-line options */
    options_description description("options");
    description.add_options()
        ("help,h", "produce help msg")
        ("pos,p", value<std::string>()->default_value("resource/base"), "positive image dir")
        ("neg,n", value<std::string>()->default_value("resource/negative"), "negative image dir")
        ("target,t", value<std::string>()->default_value("resource/target"), "detection target dir")
        ("result,r", value<std::string>()->default_value("resource/result"), "result saving dir")
        ("svm", value<std::string>()->default_value("resource/svm_traindata.xml"), "filepath to SVM XML")
        ("detector", value<std::string>()->default_value("resource/hog_detector.yml"), "filepath to HoG Detector file")
        ("no-train", bool_switch()->default_value(false), "skip flag for train")
        ("no-detection", bool_switch()->default_value(false), "skip flag for detection")
    ;

    /* Analyze commandline options */
    variables_map vm;
    try {
        store(parse_command_line(argc, argv, description), vm);
    } catch (const error_with_option_name &e) {
        std::cout << e.what() << std::endl;
    }
    notify(vm); 

    if (vm.count("help")) {
        std::cout << description << std::endl;
        return 0;
    }

    /* Retrive arguments */
    std::string positive_dir = vm["pos"].as<std::string>();
    std::string negative_dir = vm["neg"].as<std::string>();
    std::string target_dir = vm["target"].as<std::string>();
    std::string result_dir = vm["result"].as<std::string>();
    std::string svm_file = vm["svm"].as<std::string>();
    std::string detector_file = vm["detector"].as<std::string>();

    bool no_train = vm["no-train"].as<bool>();
    bool no_detection = vm["no-detection"].as<bool>();

    /* Information */
    Utils::output_log(Utils::INFO, "Positive images: " + positive_dir);
    Utils::output_log(Utils::INFO, "Negative images: " + negative_dir);
    Utils::output_log(Utils::INFO, "Target images: " + target_dir);
    Utils::output_log(Utils::INFO, "Result store: " + result_dir);
    Utils::output_log(Utils::INFO, "SVM file: " + svm_file);
    Utils::output_log(Utils::INFO, "Detector file: " + detector_file);

    /* Train process */
    if (!no_train) {
        HogSvm::train(positive_dir, negative_dir, svm_file, detector_file);
    }

    /* Detection Process */
    if (!no_detection) {
        HogSvm::detect_multiscale(target_dir, detector_file, result_dir);
    }

    return 0;
}
