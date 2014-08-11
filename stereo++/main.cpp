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
	extern int PROGRAM_ENTRY;

	switch (PROGRAM_ENTRY) {
	case 1:
		printf("\n\n========================================================\n");
		printf("PROGRAM_ENTRY = %d, invoking TestPatchMatchOnTriangles() ...\n", 1);
		printf("========================================================\n");
		void TestPatchMatchOnTriangles();
		TestPatchMatchOnTriangles();
		break;

	case 2:
		printf("\n\n========================================================\n");
		printf("PROGRAM_ENTRY = %d, invoking TestLBPOnGridGraph() ...\n", 2);
		printf("========================================================\n");
		void TestLBPOnGridGraph();
		TestLBPOnGridGraph();
		break;

	case 3:
		printf("\n\n========================================================\n");
		printf("PROGRAM_ENTRY = %d, invoking TestLBPOnFactorGraph() ...\n", 3);
		printf("========================================================\n");
		void TestLBPOnFactorGraph();
		TestLBPOnFactorGraph();
		break;
	}
	
	return 0;
}

