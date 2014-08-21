#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

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






#if 1

static void PrintProgramEntryHeader(std::string methodName, int id)
{
	printf("\n\n========================================================\n");
	printf("PROGRAM_ENTRY = %d, invoking %s() ...\n", id, methodName.c_str());
	printf("========================================================\n");
}

std::string runId;

int main(int argc, char** argv)
{
	srand(12345);


	extern int PROGRAM_ENTRY;

	switch (PROGRAM_ENTRY) {
	case 1:
		PrintProgramEntryHeader("TestPatchMatchOnTriangles", 1);
		void TestPatchMatchOnTriangles();
		TestPatchMatchOnTriangles();
		break;

	case 2:
		PrintProgramEntryHeader("TestLBPOnGridGraph", 2);
		void TestLBPOnGridGraph();
		TestLBPOnGridGraph();
		break;

	case 3:
		PrintProgramEntryHeader("TestLBPOnFactorGraph", 3);
		void TestLBPOnFactorGraph();
		TestLBPOnFactorGraph();
		break;

	case 4:
		PrintProgramEntryHeader("TestPatchMatchOnPixels", 4);
		void TestPatchMatchOnPixels();
		TestPatchMatchOnPixels();
		break;

	case 5:
		PrintProgramEntryHeader("TestSplittingPostProcess", 5);
		void TestSplittingPostProcess();
		TestSplittingPostProcess();
		break;

	case 6:
		PrintProgramEntryHeader("TestMRFStereoByGraphCut", 6);
		void TestMRFStereoByGraphCut();
		TestMRFStereoByGraphCut();
		break;

	case 7:
		PrintProgramEntryHeader("TestGroundTruthPlaneStatistics", 7);
		void TestGroundTruthPlaneStatistics();
		TestGroundTruthPlaneStatistics();
		break;

	case 8:
		PrintProgramEntryHeader("TestARAP", 8);
		void TestARAP();
		TestARAP();
		break;
	}
	
	return 0;
}
#endif