/*
 * hogsvm.hpp
 */
#ifndef _HOGSVM_HPP_
#define _HOGSVM_HPP_

#include <iostream>
#include <string>
#include <vector>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"

using namespace cv;

namespace HogSvm {
    void compute_hog(const std::string &path, std::vector<Mat> &hogs);
    void hogs2mat(const std::vector<Mat> &hogs, Mat &train_data);
    void create_trainset(std::string positive_dir, std::string negative_dir, Mat &trainset, std::vector<int> &labels);
    void svm_train(Mat &trainset, std::vector<int> &labels, std::string svm_filename);
    std::vector<float> svm2detector(Ptr<ml::SVM> &svm);
    void create_hogdetector(std::string svm_file, std::string detector_file);
    void detect_multiscale(std::string target_dir, std::string detector_file, std::string result_dir);
}

#endif