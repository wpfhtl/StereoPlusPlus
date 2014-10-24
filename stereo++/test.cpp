//#include <stdio.h>
//#include <iostream>
//#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//
//#ifdef _DEBUG
//#pragma comment(lib, "opencv_core248d.lib")
//#pragma comment(lib, "opencv_highgui248d.lib")
//#pragma comment(lib, "opencv_imgproc248d.lib")
//#pragma comment(lib, "opencv_features2d248d.lib")
//#pragma comment(lib, "opencv_calib3d248d.lib")
//#pragma comment(lib, "opencv_video248d.lib")
//#pragma comment(lib, "opencv_flann248d.lib")
//#pragma comment(lib, "opencv_nonfree248d.lib")
//#else
//#pragma comment(lib, "opencv_core248.lib")
//#pragma comment(lib, "opencv_highgui248.lib")
//#pragma comment(lib, "opencv_imgproc248.lib")
//#pragma comment(lib, "opencv_features2d248.lib")
//#pragma comment(lib, "opencv_calib3d248.lib")
//#pragma comment(lib, "opencv_video248.lib")
//#pragma comment(lib, "opencv_flann248.lib")
//#pragma comment(lib, "opencv_nonfree248.lib")
//
//#endif
//
//#include "daisy/daisy.h"
//// #include "kutility/kutility.h"
//#include "ReleaseAssert.h"
//
//
//using namespace kutility;
//
//
//
//using namespace cv;
//
//void readme();
//
//
//
///** @function readme */
//void readme()
//{
//	std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl;
//}
//
//
//
//
//
///** @function main */
//int main(int argc, char** argv)
//{
//	if (argc != 8)
//	{
//		readme(); return -1;
//	}
//
//	cv::Mat imL = cv::imread(argv[1]);
//	cv::Mat imR = cv::imread(argv[2]);
//	Rectification(imL, imR);
//	return -1;
//
//	Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
//	Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
//
//	
//
//	if (!img_1.data || !img_2.data)
//	{
//		std::cout << " --(!) Error reading images " << std::endl; return -1;
//	}
//
//	int maxCorners = atoi(argv[3]);
//	float qualityLevel = atof(argv[4]);
//	float minDistance = atof(argv[5]);
//	int blockSize = atoi(argv[6]);
//	float k = atof(argv[7]);
//
//	//cv::GoodFeaturesToTrackDetector detector(2000, 0.01, 1.0, 3, true, 0.04);
//	cv::GoodFeaturesToTrackDetector detector(maxCorners, qualityLevel, minDistance, blockSize, true, k);
//	//-- Step 1: Detect the keypoints using SURF Detector
//	int minHessian = 400;
//
//
//	//SurfFeatureDetector detector(minHessian);
//
//	std::vector<KeyPoint> keypoints_1, keypoints_2;
//
//
//	//cv::cornerSubPix()
//
//	/*detector.detect(img_1, keypoints_1);
//	detector.detect(img_2, keypoints_2);*/
//
//	detector.detect(imL, keypoints_1);
//	//detector.detect(imR, keypoints_2);
//
//	std::vector<cv::Point2f> subpixelKeypoints(keypoints_1.size());
//	for (int i = 0; i < keypoints_1.size(); i++) {
//		subpixelKeypoints[i] = keypoints_1[i].pt;
//	}
//	Size winSize = Size(2, 2);
//	Size zeroZone = Size(-1, -1);
//	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
//	cv::cornerSubPix(img_1, subpixelKeypoints, winSize, zeroZone, criteria);
//
//	printf("keyPoints.size() = %d\n", keypoints_1.size());
//	for (int retry = 0; retry < 100; retry++){
//		int idx = rand() % keypoints_1.size();
//		printf("(%6.2f, %6.2f)   (%6.2f, %6.2f)  angle=%.2f \n", keypoints_1[idx].pt.x, keypoints_1[idx].pt.y,
//			subpixelKeypoints[idx].x, subpixelKeypoints[idx].y, keypoints_1[idx].angle);
//	}
//
//	//-- Draw keypoints
//	Mat img_keypoints_1; Mat img_keypoints_2;
//
//	drawKeypoints(imL, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//	//drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//
//	//-- Show detected (drawn) keypoints
//	//imshow("Keypoints 1", img_keypoints_1);
//	//imshow("Keypoints 2", img_keypoints_2);
//
//
//	//waitKey(0);
//
//
//
//
//
//
//
//
//
//	ASSERT(img_1.isContinuous());
//	unsigned char *im = img_1.data;
//	int h = img_1.rows;
//	int w = img_1.cols;
//	daisy* desc = new daisy();
//	desc->set_image<unsigned char>(im, h, w);
//	desc->verbose(0); // 0,1,2,3 -> how much output do you want while running
//	//desc->set_parameters(rad, radq, thq, histq); // we use 15,3,8,8 for wide baseline stereo.
//	desc->set_parameters(15, 3, 8, 8);
//	desc->initialize_single_descriptor_mode();
//
//	int *orientations = desc->get_orientation_map();
//	printf("descriptor_size = %d\n", desc->descriptor_size());
//	float* thor = new float[desc->descriptor_size()];
//	float y = subpixelKeypoints[0].y;
//	float x = subpixelKeypoints[0].x;
//	int yy = y + 0.5;
//	int xx = x + 0.5;
//	//int ori = orientations[yy * w + xx];
//	int ori = 0;
//
//	float orientation = 0.f;
//	desc->get_descriptor(y, x, ori, thor); // returns normalized descriptor
//
//	delete[] thor;
//
//
//
//	return 0;
//}