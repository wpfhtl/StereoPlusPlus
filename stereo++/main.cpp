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



//int main(int argc, char **argv)
//{
//
//	if (argc != 4) {
//		printf("usage SGM.exe filePathImL filePathImR filePathDispOut\n");
//		return -1;
//	}
//
//	cv::Mat imL = cv::imread(argv[1]);
//	cv::Mat imR = cv::imread(argv[2]);
//	void RunCSGM(std::string rootFolder, cv::Mat &imL, cv::Mat &imR,
//		cv::Mat &dispL, cv::Mat &dispR, cv::Mat &validPixelMapL, cv::Mat &validPixelMapR);
//	cv::Mat dispL, dispR, validPixelL, validPixelR;
//	RunCSGM("KITTI", imL, imR, dispL, dispR, validPixelL, validPixelR);
//	dispL *= 256.0;
//	dispL.convertTo(dispL, CV_16UC1);
//	cv::imwrite(argv[3], dispL);
//
//	return 0;
//}




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

	//extern std::string ROOTFOLDER;
	//cv::Mat dispLSL = cv::imread("d:/data/stereo/" + ROOTFOLDER + "/GC+LSL.png", CV_LOAD_IMAGE_GRAYSCALE);
	//dispLSL.convertTo(dispLSL, CV_32FC1, 0.25f);
	//EvaluateDisparity(ROOTFOLDER, dispLSL, 0.5f);
	//return 0;


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

	case 9:
		PrintProgramEntryHeader("TestSemiGlobalMatching", 9);
		void TestSemiGlobalMatching();
		TestSemiGlobalMatching();
		break;

	case 10:
		PrintProgramEntryHeader("TestStereoFlow", 10);
		void TestStereoFlow();
		TestStereoFlow();
		break;

	case 11:
		PrintProgramEntryHeader("TestMySlicSegmentation", 11);
		void TestMySlicSegmentation();
		TestMySlicSegmentation();
		break;

	case 12:
		PrintProgramEntryHeader("TestSemiGlobalMatching", 12);
		void TestSemiGlobalMatching();
		TestSemiGlobalMatching();
		break;
	}
	
	return 0;
}
#endif