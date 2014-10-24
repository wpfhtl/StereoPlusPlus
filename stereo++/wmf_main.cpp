//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
//#include <iostream>
//#include <string>
//#include <sstream>
//
//#include "StereoAPI.h"
//#include "Timer.h"
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
//#endif
//
//
//
//
//int main(int argc, char **argv)
//{
//	if (argc != 5) {
//		printf("usage: %s filePathDispL filePathDispR filePathDispOut useValidPixelOnly", argv[0]);
//		exit(-1);
//	}
//
//	std::string filePathL	= std::string(argv[1]);
//	std::string filePathR	= std::string(argv[2]);
//	std::string filePathOut = std::string(argv[3]);
//	int useValidPixelOnly	= atoi(argv[4]);
//
//	cv::Mat dispL = cv::imread(filePathL, CV_LOAD_IMAGE_UNCHANGED);
//	cv::Mat dispR = cv::imread(filePathR, CV_LOAD_IMAGE_UNCHANGED);
//	dispL.convertTo(dispL, CV_32FC1, 1.f / 64.f);
//	dispR.convertTo(dispR, CV_32FC1, 1.f / 64.f);
//	cv::Mat validPixelMapL = CrossCheck(dispL, dispR, -1, 1.f);
//	bs::Timer::Tic("wmf");
//
//	bs::Timer::Toc();
//}