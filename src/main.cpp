/*
 * main.cpp
 */

#include "utils.hpp"
#include "hogsvm.hpp"
#include <boost/program_options.hpp>

int main(const int argc, const char* const * const argv) {
    using namespace boost::program_options;
    Utils::Logger logger = Utils::Logger();

    /* Definition of command-line options */
    options_description description("options");
    description.add_options()
        ("help,h", "produce help msg")
        ("dataset,d", value<std::string>()->default_value("resource/test"), "dataset directory")
        ("svm", value<std::string>()->default_value("svm_traindata.xml"), "filename to SVM XML")
        ("detector", value<std::string>()->default_value("hog_detector.yml"), "filename to HoG Detector file")
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
    std::string dataset_dir = vm["dataset"].as<std::string>();
    std::string svm_filename = vm["svm"].as<std::string>();
    std::string detector_filename = vm["detector"].as<std::string>();

    bool no_train = vm["no-train"].as<bool>();
    bool no_detection = vm["no-detection"].as<bool>();

    std::string positive_dir = dataset_dir + "/positive";
    std::string negative_dir = dataset_dir + "/negative";
    std::string target_dir = dataset_dir + "/target";
    std::string result_dir = dataset_dir + "/result";
    std::string svm_file = dataset_dir + "/" + svm_filename;
    std::string detector_file = dataset_dir + "/" + detector_filename;

    /* Information */
    logger.info("Dataset: " + dataset_dir);
    logger.debug("Positive images: " + positive_dir);
    logger.debug("Negative images: " + negative_dir);
    logger.debug("Target images: " + target_dir);
    logger.debug("Result store: " + result_dir);
    logger.info("SVM file: " + svm_file);
    logger.info("Detector file: " + detector_file);
    logger.info("Train process: " + no_train ? "false" : "true");
    logger.info("Detection process: " + no_detection ? "false" : "true");

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
