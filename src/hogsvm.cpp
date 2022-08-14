/*
 * hogsvm.cpp
 */

#include "hogsvm.hpp"
#include "utils.hpp"

using namespace cv;
namespace HogSvm {
    void compute_hog(const std::string &path, std::vector<Mat> &hogs) {
        using namespace std;
        Utils::Logger logger = Utils::Logger();
        logger.info("compute HoGs: " + path);

        /* Variables */
        Size window_sz      = Size(128, 128);
        Size block_sz       = Size(16, 16);
        Size block_str      = Size(8, 8);
        Size cell_sz        = Size(8, 8);
        int nbins           = 32;

        vector<float> descriptors;

        /* Get images in directory */
        vector<string> files = Utils::get_files(path);

        if (files.size() == 0) {
            logger.error("failed to get image files. aborting.");
            exit(1);
        }

        for (string file : files) {
            /* Load images */
            Mat img = imread(file, IMREAD_COLOR);
            logger.debug("loaded: " + file);

            /* Gray scale */
            Mat grayscale;
            cvtColor(img, grayscale, COLOR_BGR2GRAY);

            /* Downscale */
            Mat downscale;
            resize(grayscale, downscale, window_sz);
            
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

        logger.info("computed HoGs of " + std::to_string(files.size()) + " image(s)");
    }

    void compute_hog_self(const std::string path, std::vector<Mat> &hogs) {
        using namespace std;
        Utils::Logger logger = Utils::Logger();

        /* Variables */
        Size window_sz      = Size(128, 128);
        Size block_sz       = Size(16, 16);
        Size block_str      = Size(8, 8);
        Size cell_sz        = Size(8, 8);
        int nbins           = 32;

        /* Get images in directory */
        vector<string> files = Utils::get_files(path);

        if (files.size() == 0) {
            logger.error("failed to get image files. aborting.");
            exit(1);
        }

        for (string file : files) {
            /* Load images */
            Mat img = imread(file, IMREAD_COLOR);

            /* Gray scale */
            Mat grayscale;
            cvtColor(img, grayscale, COLOR_BGR2GRAY);

            /* Downscale */
            Mat downscale;
            resize(grayscale, downscale, window_sz);

            /* Get edge image */
            Mat edge_x, edge_y;
            Sobel(downscale, edge_x, CV_32F, 1, 0, 1);
            Sobel(downscale, edge_y, CV_32F, 0, 1, 1);

            /* Calc angle & magnitude of edge gradient */
            Mat magnitude, angle;
            cartToPolar(edge_x, edge_y, magnitude, angle);

            /* Quantization */
            int col = angle.cols;
            int row = angle.rows;
            for (int i = 0; i < row; i++) {
                for(int j = 0; j < col; j++) {
                    float tmp = angle.at<float>(i, j);
                    angle.at<float>(i, j) = (float)(nbins * tmp /(2 * M_PI));
                }
            }

            /* Calc histgram of each cell */
            vector<vector<vector<float>>> cell_hist;
            int cell_n_row = row / cell_sz.height;
            int cell_n_col = col / cell_sz.width;
            for (int i0 = 0; i0 < cell_n_row; i0++) {
                vector<vector<float>> cell_hist_col;
                for (int j0 = 0; j0 < cell_n_col; j0++) {
                    vector<float> histgram(nbins, 0);
                    for (int i1 = 0; i1 < cell_sz.height; i1++) {
                        for (int j1 = 0; j1 < cell_sz.width; j1++) {
                            float normalized_ang = angle.at<float>(i0 * cell_sz.height + i1, j0 * cell_sz.width + j1);
                            int bin = (int)normalized_ang;
                            float mag = magnitude.at<float>(i0 * cell_sz.height + i1, j0 * cell_sz.width + j1);
                            histgram[bin] += (1 - (normalized_ang - bin)) * mag;
                            if (bin == nbins - 1) {
                                histgram[0] += (normalized_ang - bin) * mag;
                            } else {
                                histgram[bin + 1] += (normalized_ang - bin) * mag;
                            }
                        }
                    }
                    cell_hist_col.push_back(histgram);
                }
                CV_Assert(cell_hist_col.size() == cell_n_col);
                cell_hist.push_back(cell_hist_col);
            }
            CV_Assert(cell_hist.size() == cell_n_row);

            /* Normalize histgram by block */
            int block_str_n_row = row / block_str.height;
            int block_str_n_col = col / block_str.width;
            int offset_row = block_sz.height / block_str.height;
            int offset_col = block_sz.width / block_str.width;
            int celljmp_row = block_str.height / cell_sz.height;
            int celljmp_col = block_str.width / cell_sz.width;
            int in_block_row = block_sz.height / cell_sz.height;
            int in_block_col = block_sz.width / cell_sz.width;
            int block_n_row = block_str_n_row - (offset_row - 1);
            int block_n_col = block_str_n_col - (offset_col - 1);
            double epsilon = 1.0000000000000000e-3;
            vector<vector<float>> normalized_vecs; 
            for (int i0 = 0; i0 < block_n_row; i0++) {
                for (int j0 = 0; j0 < block_n_col; j0++) {
                    float sq_sum = .0;
                    for (int i1 = 0; i1 < in_block_row; i1++) {
                        for (int j1 = 0; j1 < in_block_col; j1++) {
                            for (float val : cell_hist[i0 * celljmp_row + i1][j0 * celljmp_col + j1]) {
                                sq_sum += val * val;
                            }
                        }
                    }
                    float vec_len = sqrtf(sq_sum + epsilon);
                    vector<float> normalized_vec;
                    for (int i1 = 0; i1 < in_block_row; i1++) {
                        for (int j1 = 0; j1 < in_block_col; j1++) {
                            for (int k1 = 0; k1 < nbins; k1++) {
                                float tmp = cell_hist[i0 * celljmp_row + i1][j0 * celljmp_col + j1][k1];
                                float normalized = tmp / vec_len;
                                if (normalized > 0.2) {
                                    normalized = 0.2;
                                }
                                normalized_vec.push_back(normalized);
                            }
                        }
                    }
                    CV_Assert(normalized_vec.size() == in_block_row * in_block_col * nbins);
                    normalized_vecs.push_back(normalized_vec);
                }
            }
            CV_Assert(normalized_vecs.size() == block_n_row * block_n_col);

            /* Faltten normalized feature vector */
            vector<float> flattened_vector;
            for (vector<float> vec : normalized_vecs) {
                for (float val : vec) {
                    flattened_vector.push_back(val);
                }
            }
            CV_Assert(flattened_vector.size() == block_n_row * block_n_row * in_block_row * in_block_col * nbins);

            /* Convert to Mat & store */
            hogs.push_back(Mat(flattened_vector).clone()); 
        }
        logger.info("computed HoGs of " + std::to_string(files.size()) + " image(s)");
    }

    void hogs2mat(const std::vector<Mat> &hogs, Mat &train_data) {
        /* Variables */
        Utils::Logger logger = Utils::Logger();
        const int rows = (int) hogs.size();
        const int cols = (int) std::max(hogs[0].cols, hogs[0].rows);
        Mat tmp(1, cols, CV_32FC1);
        train_data = Mat(rows, cols, CV_32FC1);

        logger.info("convert HoGs to training data");

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

    void create_trainset(std::string positive_dir, std::string negative_dir, Mat &trainset, std::vector<int> &labels, bool self) {
        Utils::Logger logger = Utils::Logger();
        std::vector<Mat> hogs;

        logger.info("create trainset");
        
        /* compute HOGs of positive images */
        logger.info("compute HoGs of positive images");

        if (!self) {
            compute_hog(positive_dir, hogs);
        } else {
            compute_hog_self(positive_dir, hogs);
        }
        int positive_n = (int)hogs.size();
        labels.assign(positive_n, 1);

        logger.info("positive count: " + std::to_string(positive_n));

        /* compute HoGs of negative images */
        logger.info("compute HoGs of negative images");

        if (!self) {
            compute_hog(negative_dir, hogs);
        } else {
            compute_hog_self(negative_dir, hogs);
        }
        int negative_n = (int)hogs.size() - positive_n; 
        labels.insert(labels.end(), negative_n, -1);

        logger.info("negative count: " + std::to_string(negative_n));
        logger.info("total count: " + std::to_string(hogs.size()));

        /* convert trainset */
        hogs2mat(hogs, trainset);
    }

    void svm_train(Mat &trainset, std::vector<int> &labels, std::string svm_filename) {
        using namespace cv::ml;
        Utils::Logger logger = Utils::Logger();
        logger.info("proceeding SVM train");

        /* Configure SVM instance */
        Ptr<SVM> svm = SVM::create();
        svm->setType(SVM::EPS_SVR);
        svm->setKernel(SVM::LINEAR);
        svm->setCoef0(0.0);
        svm->setDegree(3);
        svm->setGamma(0);
        svm->setC(1.0);
        svm->setNu(0.5);
        svm->setP(0.1);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3));

        logger.info("instance configured");
        logger.info("starting train");

        /* Train */
        svm->train(trainset, ROW_SAMPLE, labels);
        svm->save(svm_filename);

        logger.info("complete. file saved as " + svm_filename);
    }

    std::vector<float> svm2detector(Ptr<ml::SVM> &svm) {
        Utils::Logger logger = Utils::Logger();
        logger.info("extract detector from SVM");

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
        Utils::Logger logger = Utils::Logger();
        logger.info("creating HoG detector from SVM");

        /* Load SVM from XML */
        Ptr<SVM> svm = SVM::load(svm_file);
        logger.info("SVM loaded from " + svm_file);

        HOGDescriptor hog;
        Size window_sz      = Size(128, 128);
        Size block_sz       = Size(16, 16);
        Size block_str      = Size(8, 8);
        Size cell_sz        = Size(8, 8);
        int nbins           = 32;
        hog.winSize     = window_sz;
        hog.blockSize   = block_sz;
        hog.blockStride = block_str;
        hog.cellSize    = cell_sz;
        hog.nbins       = nbins;
        hog.setSVMDetector(svm2detector(svm));
        hog.save(detector_file);

        logger.info("detector saved as " + detector_file);
    }
    
    void detect_multiscale(std::string target_dir, std::string detector_file, std::string result_dir) {
        Utils::Logger logger = Utils::Logger();
        logger.info("[Start detection: Multiscale]");
        logger.info("\ttarget: " + target_dir);
        logger.info("\tdetector file: " + detector_file);
        logger.info("\tresults saved to " + result_dir);

        /* Load detector */
        HOGDescriptor hog;
        hog.load(detector_file);
        logger.info("detector loaded");

        /* Retrive detection targets */
        std::vector<std::string> files = Utils::get_files(target_dir);
        if (files.size() == 0) {
            logger.error("failed to get image files. aborting.");
            exit(1);
        }

        /* Proceed detection */
        logger.info(std::to_string(files.size()) + "file(s) in detection process");
        logger.info("proceeding...");
        int idx = 0;
        for (std::string file : files) {
            Mat img = imread(file, IMREAD_COLOR);
            logger.debug("loaded: " + file);

            std::vector<Rect> detections;
            std::vector<double> found_weights;
            hog.detectMultiScale(
                img,
                detections,
                found_weights
            );
            logger.debug(std::to_string(detections.size()) + " object(s) detected");

            /* Create & save result */
            for (size_t i = 0; i < detections.size(); i++) {
                logger.debug("\tobject " + std::to_string(i) + " weight: " + std::to_string(found_weights[i]));
                if (found_weights[i] * found_weights[i] < 0.1) {
                    logger.debug("\t\tlow weight. skipping.");
                    continue;
                }
                Scalar color = Scalar(0., found_weights[i] * found_weights[i] * 200.0, 0.);
                rectangle(img, detections[i], color, img.cols / 400 + 1);
            }
            std::string outimg = result_dir + "/res_" + std::to_string(idx) + ".png";
            imwrite(outimg, img);
            logger.debug("image saved: " + outimg);
            idx++;
        }
        logger.info("finished");
        logger.info("results saved into " + result_dir);
    }

    void train(std::string positive_dir, std::string negative_dir, std::string svm_file, std::string detector_file, bool self) {
        Mat trainset;
        std::vector<int> labels;
        create_trainset(positive_dir, negative_dir, trainset, labels, self);
        svm_train(trainset, labels, svm_file);
        create_hogdetector(svm_file, detector_file);
    }
}