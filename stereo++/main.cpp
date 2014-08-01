#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "StereoAPI.h"

#if 0
int main()
{

	cv::Mat imL = cv::imread("D:/data/stereo/teddy/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/teddy/im6.png");
	cv::Mat dispL, dispR;

	//RunPatchMatchOnPixels("teddy", imL, imR, dispL, dispR);
	void RunPatchMatchOnPixels3(std::string rootFolder, cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR);
	RunPatchMatchOnPixels("teddy", imL, imR, dispL, dispR);
}
#endif

int main()
{
	/*void TestTriangulation2D();
	TestTriangulation2D();*/

	void TestPatchMatchOnTriangles();
	TestPatchMatchOnTriangles();

	return 0;
}

