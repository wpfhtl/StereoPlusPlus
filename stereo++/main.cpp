#if 1
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <sstream>

#include "StereoAPI.h"
#include "Timer.h"

#ifdef _DEBUG
#pragma comment(lib, "opencv_core248d.lib")
#pragma comment(lib, "opencv_highgui248d.lib")
#pragma comment(lib, "opencv_imgproc248d.lib")
#pragma comment(lib, "opencv_features2d248d.lib")
#pragma comment(lib, "opencv_calib3d248d.lib")
#pragma comment(lib, "opencv_video248d.lib")
#pragma comment(lib, "opencv_flann248d.lib")
#pragma comment(lib, "opencv_nonfree248d.lib")
#else
#pragma comment(lib, "opencv_core248.lib")
#pragma comment(lib, "opencv_highgui248.lib")
#pragma comment(lib, "opencv_imgproc248.lib")
#pragma comment(lib, "opencv_features2d248.lib")
#pragma comment(lib, "opencv_calib3d248.lib")
#pragma comment(lib, "opencv_video248.lib")
#pragma comment(lib, "opencv_flann248.lib")
#pragma comment(lib, "opencv_nonfree248.lib")
#endif




std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

int ReadCalibFile(std::string filePath)
{
	FILE *fid = fopen(filePath.c_str(), "r");
	char lineBuf[1024];
	char *line;
	int numDisps = -1;

	while ((line = fgets(lineBuf, 1023, fid)) != NULL) {
		std::vector<std::string> tokens = split(line, '=');
		if (tokens.size() > 0 && tokens[0] == "ndisp") {
			numDisps = atoi(tokens[1].c_str());
			break;
		}
	}
	fclose(fid);

	if (numDisps == -1) {
		printf("Fail to read calib file, exiting...\n");
		exit(-1);
	}
	
	return numDisps;
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		printf("usage: .exe Midd3 midd3Resolution midd3TestCaseId numRegions DO_EVAL VISUALIZE_EVAL outImagePath\n");
		printf("usage: .exe Midd3 kittiTestCaseId numRegions DO_EVAL VISUALIZE_EVAL outImagePath [leftOraclePath rightOraclePath]\n");
		exit(-1);
	}

	extern int DO_EVAL;
	extern int VISUALIZE_EVAL;
	extern int NUM_PREFERED_REGIONS;;
	extern int gNumDisps;
	std::string filePathImageL, filePathImageR, filePathOutImage;

	std::string benchmark = argv[1];
	if (benchmark == "Midd3") {
		extern std::string midd3Resolution;
		extern std::string midd3TestCaseId;
		bool FileExist(std::string filePath);

		midd3Resolution			= argv[2];
		midd3TestCaseId			= argv[3];
		NUM_PREFERED_REGIONS	= atoi(argv[4]);
		DO_EVAL					= atoi(argv[5]);
		VISUALIZE_EVAL			= atoi(argv[6]);
		filePathOutImage		= argv[7];
		filePathImageL = "D:\\data\\MiddEval3\\" + midd3Resolution + "\\" + midd3TestCaseId + "\\im0.png";
		filePathImageR = "D:\\data\\MiddEval3\\" + midd3Resolution + "\\" + midd3TestCaseId + "\\im1_rectified.png";
		
		if (!FileExist(filePathImageR)) {
			filePathImageR = "D:\\data\\MiddEval3\\" + midd3Resolution + "\\" + midd3TestCaseId + "\\im1.png";
		}

		gNumDisps = ReadCalibFile("D:\\data\\MiddEval3\\" + midd3Resolution + "\\" + midd3TestCaseId + "\\calib.txt");
	}
	if (benchmark == "KITTI") {
		extern std::string kittiTestCaseId;

		kittiTestCaseId			= argv[2];
		NUM_PREFERED_REGIONS	= atoi(argv[3]);
		DO_EVAL					= atoi(argv[4]);
		VISUALIZE_EVAL			= atoi(argv[5]);
		filePathOutImage		= argv[6];
		if (argc == 9) {
			extern std::string gFilePathOracleL, gFilePathOracleR;
			gFilePathOracleL	= argv[7];
			gFilePathOracleR	= argv[8];
		}
		filePathImageL = "D:\\data\\KITTI\\training\\colored_0\\" + kittiTestCaseId + ".png";
		filePathImageR = "D:\\data\\KITTI\\training\\colored_1\\" + kittiTestCaseId + ".png";

		gNumDisps = 256;
	}

	
	cv::Mat imL = cv::imread(filePathImageL);
	cv::Mat imR = cv::imread(filePathImageR);
	cv::Mat dispL, dispR;

#if 1
	void RunARAP(std::string rootFolder, cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR,
		std::string filePathImageL, std::string filePathImageR, std::string filePathOutImage);

	bs::Timer::Tic("ARAP");
	RunARAP(benchmark, imL, imR, dispL, dispR, filePathImageL, filePathImageR, filePathOutImage);
	bs::Timer::Toc();
#endif

#if 0
	void YamaguchiSGM(std::string benchmark, std::string filePathImageL, std::string filePathImageR, std::string filePathImageOut);
	bs::Timer::Tic("YamaguchiSGM");
	YamaguchiSGM(benchmark, filePathImageL, filePathImageR, filePathOutImage);
	bs::Timer::Toc();
#endif 

#if 0
	void RunPatchMatchOnPixels(std::string rootFolder, cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR, 
		std::string filePathImageL, std::string filePathImageR, std::string filePathOutImage);
	bs::Timer::Tic("PatchMatchOnPixels");
	RunPatchMatchOnPixels(benchmark, imL, imR, dispL, dispR, filePathImageL, filePathImageR, filePathOutImage);
	bs::Timer::Toc();
#endif 

#if 0
	extern std::string midd3TestCaseId;
	std::string filePathDispL = "D:\\data\\MiddEval3\\myresults\\PatchMatchOnPixels\\trainingH\\" + midd3TestCaseId + ".png_dispL.png";
	std::string filePathDispR = "D:\\data\\MiddEval3\\myresults\\PatchMatchOnPixels\\trainingH\\" + midd3TestCaseId + ".png_dispR.png";
	dispL = cv::imread(filePathDispL, CV_LOAD_IMAGE_UNCHANGED);
	dispR = cv::imread(filePathDispR, CV_LOAD_IMAGE_UNCHANGED);

	dispL.convertTo(dispL, CV_32FC1, 1.f / 64.f);
	dispR.convertTo(dispR, CV_32FC1, 1.f / 64.f);

	cv::Mat validMapL = CrossCheck(dispL, dispR, -1, 1.f);
	void FastWeightedMedianFilterInvalidPixels(cv::Mat &disp, cv::Mat &validPixelMap, cv::Mat &img);
	bs::Timer::Tic("wmf");
	FastWeightedMedianFilterInvalidPixels(dispL, validMapL, imL);
	bs::Timer::Toc();

	std::string filePathWmfNewL = "D:\\data\\MiddEval3\\myresults\\PatchMatchOnPixels\\trainingH\\" + midd3TestCaseId + ".png_wmfNewL.png";
	dispL.convertTo(dispL, CV_16UC1, 64.f);
	cv::imwrite(filePathWmfNewL, dispL);
#endif

#if 0
	void Rectification(cv::Mat &imL, cv::Mat &imR);
	Rectification(imL, imR);
#endif 

}


#endif




#if 0
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