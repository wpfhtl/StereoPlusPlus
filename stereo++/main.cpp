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


int main()
{
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

