#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "StereoAPI.h"

#ifdef _DEBUG
#pragma comment(lib, "opencv_core248d.lib")
#pragma comment(lib, "opencv_highgui248d.lib")
#pragma comment(lib, "opencv_imgproc248d.lib")
#pragma comment(lib, "opencv_features2d248d.lib")
#pragma comment(lib, "opencv_calib3d248d.lib")
#pragma comment(lib, "opencv_video248d.lib")
#pragma comment(lib, "opencv_flann248d.lib")
#else
#pragma comment(lib, "opencv_core248.lib")
#pragma comment(lib, "opencv_highgui248.lib")
#pragma comment(lib, "opencv_imgproc248.lib")
#pragma comment(lib, "opencv_features2d248.lib")
#pragma comment(lib, "opencv_calib3d248.lib")
#pragma comment(lib, "opencv_video248.lib")
#pragma comment(lib, "opencv_flann248.lib")
#endif


#include <iostream>

int main()
{
	//cv::Mat mat(3, 3, CV_32FC1);
	//mat.setTo(3.3);
	//std::cout << mat << "\n";
	////for (int y = 0; y < 3; y++) {
	////	for (int x = 0; x < 3; x++) {
	////		printf("%d ....... ", mat.at<bool>(y, x));
	////		printf("%d\n", mat.at<int>(y, x));
	////		/*if (!mat.at<bool>(y, x)) {
	////			printf("BUG!! at (%d, %d)\n", y, x);
	////		}*/
	////	}
	////}
	//return 0;

	//void TestPatchMatchOnPixels();
	//TestPatchMatchOnPixels();

	//void TestTriangulation2D();
	//TestTriangulation2D();

	void TestPatchMatchOnTriangles();
	TestPatchMatchOnTriangles();

	//void TestLBPOnGridGraph();
	//TestLBPOnGridGraph();

	//void TestLBPOnFactorGraph();
	//TestLBPOnFactorGraph();

	return 0;
}

