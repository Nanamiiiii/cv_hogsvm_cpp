/*
 * hogsvm.cpp
 */

#include "hogsvm.hpp"
#include "utils.hpp"

using namespace cv;
namespace HogSvm {
    void compute_hog(const std::string &path, std::vector<Mat> &hogs) {
        using namespace std;
        Utils::output_log(Utils::INFO, "compute HoGs: " + path);

        /* Variables */
        Size window_sz      = Size(64, 64);
        Size block_sz       = Size(16, 16);
        Size block_str      = Size(4, 4);
        Size cell_sz        = Size(4, 4);
        int nbins           = 9;

        vector<float> descriptors;

        /* Get images in directory */
        vector<string> files = Utils::get_files(path);

        if (files.size() == 0) {
            Utils::output_log(Utils::ERR, "failed to get image files. aborting.");
            exit(1);
        }

        for (string file : files) {
            /* Load images */
            Mat img = imread(file, IMREAD_COLOR);
            Utils::output_log(Utils::DEBUG, "loaded: " + file);

            /* Gray scale */
            Mat grayscale;
            cvtColor(img, grayscale, COLOR_BGR2GRAY);

            /* Downscale 64x64 */
            Mat downscale;
            resize(grayscale, downscale, Size(64, 64));
            
            /* Create HOGDescriptor */
            HOGDescriptor hog = HOGDescriptor(
                window_sz,
                block_sz,
                block_str,
                cell_sz,
                nbins
            );

            /* Compute hog */
            hog.compute(downscale, descriptors);
            hogs.push_back(Mat(descriptors).clone());
        }

        Utils::output_log(Utils::INFO, "computed HoGs of " + std::to_string(files.size()) + " image(s)");
    }

    void hogs2mat(const std::vector<Mat> &hogs, Mat &train_data) {
        /* Variables */
        const int rows = (int) hogs.size();
        const int cols = (int) std::max(hogs[0].cols, hogs[0].rows);
        Mat tmp(1, cols, CV_32FC1);
        train_data = Mat(rows, cols, CV_32FC1);

        Utils::output_log(Utils::INFO, "convert HoGs to training data");

        /* Convert to single Mat */
        for (size_t i = 0; i < hogs.size(); i++) {
            CV_Assert(hogs[i].cols == 1 || hogs[i].rows == 1);
            if (hogs[i].cols == 1) {
                transpose(hogs[i], tmp);
                tmp.copyTo(train_data.row((int)i));
            } else if (hogs[i].rows == 1) {
                hogs[i].copyTo(train_data.row((int)i));
            }
        }
    }

    void create_trainset(std::string positive_dir, std::string negative_dir, Mat &trainset, std::vector<int> &labels) {
        std::vector<Mat> hogs;

        Utils::output_log(Utils::INFO, "create trainset");
        
        /* compute HOGs of positive images */
        Utils::output_log(Utils::INFO, "compute HoGs of positive images");

        compute_hog(positive_dir, hogs);
        int positive_n = hogs.size();
        labels.assign(positive_n, 1);

        Utils::output_log(Utils::INFO, "positive count: " + std::to_string(positive_n));

        /* compute HoGs of negative images */
        Utils::output_log(Utils::INFO, "compute HoGs of negative images");

        compute_hog(negative_dir, hogs);
        int negative_n = hogs.size() - positive_n; 
        labels.insert(labels.end(), negative_n, -1);

        Utils::output_log(Utils::INFO, "negative count: " + std::to_string(negative_n));
        Utils::output_log(Utils::INFO, "total count: " + std::to_string(hogs.size()));

        /* convert trainset */
        hogs2mat(hogs, trainset);
    }

    void svm_train(Mat &trainset, std::vector<int> &labels, std::string svm_filename) {
        using namespace cv::ml;
        Utils::output_log(Utils::INFO, "proceeding SVM train");

        /* Configure SVM instance */
        Ptr<SVM> svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::LINEAR);
        svm->setGamma(1.0);
        svm->setC(1.0);
        svm->setTermCriteria(TermCriteria(TermCriteria::COUNT, 100, 1e-6));

        Utils::output_log(Utils::INFO, "instance configured");
        Utils::output_log(Utils::INFO, "starting train");

        /* Train */
        svm->train(trainset, ROW_SAMPLE, labels);
        svm->save(svm_filename);

        Utils::output_log(Utils::INFO, "complete. file saved as " + svm_filename);
    }

    std::vector<float> svm2detector(Ptr<ml::SVM> &svm) {
        Utils::output_log(Utils::INFO, "extract detector from SVM");

        /* Export support vector */
        Mat sv = svm->getSupportVectors();
        const int sv_num = sv.rows;

        /* Export decision function */
        Mat alpha, svidx;
        double rho = svm->getDecisionFunction(0, alpha, svidx);

        CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_num == 1);
        CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
                  (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
        CV_Assert(sv.type() == CV_32F);

        /* Forming detector */
        std::vector<float> detector(sv.cols + 1);
        memcpy(&detector[0], sv.ptr(), sv.cols * sizeof(detector[0]));
        detector[sv.cols] = (float)-rho;
        return detector;
    }

    void create_hogdetector(std::string svm_file, std::string detector_file) {
        using namespace cv::ml;
        Utils::output_log(Utils::INFO, "creating HoG detector from SVM");

        /* Load SVM from XML */
        Ptr<SVM> svm = SVM::load(svm_file);
        Utils::output_log(Utils::INFO, "SVM loaded from " + svm_file);

        HOGDescriptor hog;
        hog.winSize = Size(64, 64);
        hog.setSVMDetector(svm2detector(svm));
        hog.save(detector_file);

        Utils::output_log(Utils::INFO, "detector saved as " + detector_file);
    }
    
    void detect_multiscale(std::string target_dir, std::string detector_file, std::string result_dir) {
        Utils::output_log(Utils::INFO, "[Start detection: Multiscale]");
        Utils::output_log(Utils::INFO, "\ttarget: " + target_dir);
        Utils::output_log(Utils::INFO, "\tdetector file: " + detector_file);
        Utils::output_log(Utils::INFO, "\tresults saved to " + result_dir);

        /* Load detector */
        HOGDescriptor hog;
        hog.load(detector_file);
        Utils::output_log(Utils::DEBUG, "detector loaded");

        /* Retrive detection targets */
        std::vector<std::string> files = Utils::get_files(target_dir);
        if (files.size() == 0) {
            Utils::output_log(Utils::ERR, "failed to get image files. aborting.");
            exit(1);
        }

        /* Proceed detection */
        Utils::output_log(Utils::INFO, std::to_string(files.size()) + "file(s) in detection process");
        Utils::output_log(Utils::INFO, "proceeding...");
        int idx = 0;
        for (std::string file : files) {
            Mat img = imread(file, IMREAD_COLOR);
            Utils::output_log(Utils::DEBUG, "loaded: " + file);

            std::vector<Rect> detections;
            std::vector<double> found_weights;
            hog.detectMultiScale(
                img,
                detections,
                found_weights
            );
            Utils::output_log(Utils::DEBUG, std::to_string(detections.size()) + " object(s) detected");

            /* Create & save result */
            for (size_t i = 0; i < detections.size(); i++) {
                Scalar color = Scalar(0., found_weights[i] * found_weights[i] * 200.0, 0.);
                rectangle(img, detections[i], color, img.cols / 400 + 1);
            }
            std::string outimg = result_dir + "res_" + std::to_string(idx) + ".png";
            imwrite(outimg, img);
            Utils::output_log(Utils::DEBUG, "image saved: " + outimg);
        }
        Utils::output_log(Utils::INFO, "finished");
        Utils::output_log(Utils::INFO, "results saved into " + result_dir);
    }
}