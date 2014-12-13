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


//int main(int argc, char **argv)
//{
//	void TestStereoRectification();
//	TestStereoRectification();
//	return 0;
//}



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

#if 0
using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

/**
* @function CannyThreshold
* @brief Trackbar callback - Canny thresholds input with a ratio 1:3
*/
void CannyThreshold(int, void*)
{
	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	src.copyTo(dst, detected_edges);
	imshow(window_name, dst);
}


/** @function main */
int main(int argc, char** argv)
{
	/// Load an image
	src = imread(argv[1]);

	if (!src.data)
	{
		return -1;
	}

	/// Create a matrix of the same type and size as src (for dst)
	dst.create(src.size(), src.type());

	/// Convert the image to grayscale
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Create a window
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create a Trackbar for user to enter threshold
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

	/// Show the image
	CannyThreshold(0, 0);

	/// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
}
#endif
#if 0
int main(int argc, char **argv)
{
	//void ImageDomainTessellation(cv::Mat &img, std::vector<cv::Point2f> &vertexCoordList,
	//	std::vector<std::vector<int>> &triVertexIndsList);
	//cv::Mat img = cv::imread("d:/data/stereo/teddy/im2.png");
	//std::vector<cv::Point2f> vertexCoodList;
	//std::vector<std::vector<int>> triVertexIndsList;
	//ImageDomainTessellation(img, vertexCoodList, triVertexIndsList);
	//return -1;

	if (argc < 2) {
		printf("usage: .exe filePathStereoParams Midd3 midd3Resolution midd3TestCaseId numRegions DO_EVAL VISUALIZE_EVAL outImagePath\n");
		printf("usage: .exe filePathStereoParams KITTI kittiTestCaseId numRegions DO_EVAL VISUALIZE_EVAL outImagePath [leftOraclePath rightOraclePath]\n");
		printf("usage: .exe filePathStereoParams Herodion numDisps regions leftPath rightpath outPath");
		exit(-1);
	}

	void ReadStereoParameters(std::string filePathStereoParams);
	std::string filePathStereoParams = argv[1];
	if (filePathStereoParams == "NULL") {
		filePathStereoParams = "d:/data/stereo_params.txt";
	}
	ReadStereoParameters(filePathStereoParams);


	extern int DO_EVAL;
	extern int VISUALIZE_EVAL;
	extern int NUM_PREFERED_REGIONS;;
	extern int gNumDisps;
	std::string filePathImageL, filePathImageR, filePathOutImage;
 
	std::string benchmark = argv[2];
	if (benchmark == "Midd3") {
		extern std::string midd3Resolution;
		extern std::string midd3TestCaseId;
		extern std::string midd3BasePath;
		bool FileExist(std::string filePath);

		midd3Resolution			= argv[3];
		midd3TestCaseId			= argv[4];
		NUM_PREFERED_REGIONS	= atoi(argv[5]);
		DO_EVAL					= atoi(argv[6]);
		VISUALIZE_EVAL			= atoi(argv[7]);
		filePathOutImage		= argv[8];

		filePathImageL = midd3BasePath + "\\" + midd3Resolution + "\\" + midd3TestCaseId + "\\im0.png";
		filePathImageR = midd3BasePath + "\\" + midd3Resolution + "\\" + midd3TestCaseId + "\\im1_rectified.png";
		
		if (!FileExist(filePathImageR)) {
			filePathImageR = midd3BasePath + "\\" + midd3Resolution + "\\" + midd3TestCaseId + "\\im1.png";
		}

		gNumDisps = ReadCalibFile(midd3BasePath + "\\" + midd3Resolution + "\\" + midd3TestCaseId + "\\calib.txt");
	}
	if (benchmark == "Midd2") {
		extern std::string midd3Resolution;
		extern std::string midd3TestCaseId;
		extern std::string midd3BasePath;
		bool FileExist(std::string filePath);

		midd3Resolution = argv[3];
		midd3TestCaseId = argv[4];
		NUM_PREFERED_REGIONS = atoi(argv[5]);
		DO_EVAL = atoi(argv[6]);
		VISUALIZE_EVAL = atoi(argv[7]);
		filePathOutImage = argv[8];

		filePathImageL = "d:/data/midd2/thirdSize/" + midd3TestCaseId + "/view1.png";
		filePathImageR = "d:/data/midd2/thirdSize/" + midd3TestCaseId + "/view5.png";

		gNumDisps = 80;
	}
	if (benchmark == "KITTI") {
		extern std::string kittiBasePath;
		extern std::string kittiTestCaseId;

		kittiTestCaseId			= argv[3];
		NUM_PREFERED_REGIONS	= atoi(argv[4]);
		DO_EVAL					= atoi(argv[5]);
		VISUALIZE_EVAL			= atoi(argv[6]);
		filePathOutImage		= argv[7];
		if (argc == 9) {
			extern std::string gFilePathOracleL, gFilePathOracleR;
			gFilePathOracleL	= argv[8];
			gFilePathOracleR	= argv[9];
		}
		filePathImageL = kittiBasePath + "\\training\\colored_0\\" + kittiTestCaseId + ".png";
		filePathImageR = kittiBasePath + "\\training\\colored_1\\" + kittiTestCaseId + ".png";

		gNumDisps = 256;
	} 
	if (benchmark == "Herodion") {
		gNumDisps				= atoi(argv[3]);
		NUM_PREFERED_REGIONS	= atoi(argv[4]);
		filePathImageL			= argv[5];
		filePathImageR			= argv[6];
		filePathOutImage		= argv[7];
		DO_EVAL = 0;
		VISUALIZE_EVAL = 0;
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

#endif



std::string runId;
float RENDER_ALPHA;
int USE_GT_DISP = 0;
std::string filePathRenderOutput;
#if 0
static void PrintProgramEntryHeader(std::string methodName, int id)
{
	printf("\n\n========================================================\n");
	printf("PROGRAM_ENTRY = %d, invoking %s() ...\n", id, methodName.c_str());
	printf("========================================================\n");
}



void CombineVirtualView();

int main(int argc, char** argv)
{
	//if (argc != 3) {
	//	printf("usage: .exe filePathImageIn filePathImageOut\n");
	//	exit(-1);
	//}
	//void FillRenderHoles(std::string filePathImageIn, std::string filePathImageOut);
	//FillRenderHoles(argv[1], argv[2]);
	//return 0;

	
	//void PutPSNROnImage(int x, int y);
	//int x = atoi(argv[1]);
	//int y = atoi(argv[2]);
	//PutPSNROnImage(x, y);
	//return -1;
	//filePathRenderOutput = "herodion.png";
	//CombineVirtualView();
	//return -1;

	void ReadStereoParameters(std::string filePathStereoParams);
	std::string filePathStereoParams = "d:/data/stereo_params.txt";
	ReadStereoParameters(filePathStereoParams);

	if (argc >= 2) {
		extern std::string ROOTFOLDER;
		ROOTFOLDER = argv[1];
	}

	if (argc >= 3) {
		RENDER_ALPHA = atof(argv[2]);
	}
	else {
		RENDER_ALPHA = 0.5;
	}
	printf("RENDER_ALPHA = %f\n", RENDER_ALPHA);

	if (argc >= 4) {
		USE_GT_DISP = atoi(argv[3]);
	}

	if (argc >= 5) {
		filePathRenderOutput = argv[4];
	}
	
	//extern std::string ROOTFOLDER;
	//cv::Mat dispLSL = cv::imread("d:/data/stereo/" + ROOTFOLDER + "/GC+LSL.png", CV_LOAD_IMAGE_GRAYSCALE);
	//dispLSL.convertTo(dispLSL, CV_32FC1, 0.25f);
	//EvaluateDisparity(ROOTFOLDER, dispLSL, 0.5f);
	//return 0;

	void RenderByMyself();
	//RenderByMyself();
	int RenderByOpenGL();
	//RenderByOpenGL();
	//exit(1);
	


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


static cv::Mat MYDISP;

void OnMouseCombine(int event, int x, int y, int flags, void *param)
{
	//cv::Mat &canvas = *(cv::Mat*)((void**)param)[0];
	cv::Mat &canvas = *(cv::Mat*)param;


	int numRows = canvas.rows, numCols = canvas.cols;
	numCols /= 2;

	cv::Mat &GT = canvas(cv::Rect(0, 0, numCols, numRows));
	cv::Mat &MY = canvas(cv::Rect(numCols, 0, numCols, numRows));
	x %= numCols;
	y %= numRows;

	if (MYDISP.empty()) {
		MYDISP = MY.clone();
	}


	if (event == CV_EVENT_LBUTTONDOWN)
	{
		const int stride = 15;
		for (int yy = y - stride; yy <= y + stride; yy++) {
			for (int xx = x - stride; xx <= x + stride; xx++) {
				if (InBound(yy, xx, numRows, numCols)) {
					MY.at<cv::Vec3b>(yy, xx) = GT.at<cv::Vec3b>(yy, xx);
				}
			}
		}

		cv::imshow("canvas", canvas);
	}

	if (event == CV_EVENT_RBUTTONDOWN)
	{
		const int stride = 15;
		for (int yy = y - stride; yy <= y + stride; yy++) {
			for (int xx = x - stride; xx <= x + stride; xx++) {
				if (InBound(yy, xx, numRows, numCols)) {
					MY.at<cv::Vec3b>(yy, xx) = MYDISP.at<cv::Vec3b>(yy, xx);
				}
			}
		}

		cv::imshow("canvas", canvas);
	}

	if (event == CV_EVENT_LBUTTONDBLCLK) {
		cv::imwrite("d:/modified.png", MY);
		printf("image save.\n");
	}
}

void CombineVirtualView()
{
	/*std::string filePathGT = "D:/data/CVPR_figures/Books/GT.png";
	std::string filePathMY = "D:/data/CVPR_figures/Books/Ours.png";*/

	std::string filePathGT = "D:/data/Herodion/camera6_138R.png";
	std::string filePathMY = "herodion_24.57.png";

	//std::string filePathGT = "D:/data/CVPR_figures/Reindeer/GT.png";
	//std::string filePathMY = "D:/data/CVPR_figures/Reindeer/ours.png";

	cv::Mat imgGT = cv::imread(filePathGT);
	cv::Mat imgMY = cv::imread(filePathMY);

	cv::Mat canvas;
	cv::hconcat(imgGT, imgMY, canvas);
	cv::imshow("canvas", canvas);
	cv::setMouseCallback("canvas", OnMouseCombine, &canvas);
	cv::waitKey(0);
}

void PutPSNROnImage(int dd, int ss)
{
	std::string filePath = "d:/modified.png";
	cv::Mat img = cv::imread(filePath);
	int numRows = img.rows, numCols = img.cols;
	char text[1024];
	sprintf(text, "%.2f", 25.66f);

	float standardLen = 463.f;
	float x = 190;
	float y = 60;
	x *= (numCols / standardLen);
	y *= (numCols / standardLen);
	int thickness = 6 * (numCols / standardLen) + 0.5;
	float scale = numCols / standardLen * 2;


	cv::putText(img, std::string(text), cv::Point2f(numCols - x, y), 0, scale, cv::Scalar(0, 0, 255, 1), thickness, CV_AA);
	cv::imshow("img", img);
	cv::imwrite("d:/PSNR.png", img);
	cv::waitKey(0);

	/*std::string folders[] = { "Books", "Reindeer", "Herodion_Render" };
	std::string methodNames[] = { "Volume", "TreeFiltering", "Fickel", "Ours" };
	float standardLen = 463.f;

	float PSNR[3][4] = { { 24.2,
		25.83,
		28.41,
		30.07},
		{ 25.03,
		26.03,
		27.7,
		31.97 },
		{
			21.997,	22.0508,	24.2544,	25.6591

		}
	};


	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			std::string filePath = "C:/Users/v-cz/Dropbox/CVPR'15/RenderResults/" + folders[i] + "/" + methodNames[j] + ".png";
			float psnr = PSNR[i][j];

			cv::Mat img = cv::imread(filePath);
			int numRows = img.rows, numCols = img.cols;
			char text[1024];
			sprintf(text, "%.2f", psnr);

			float x = 190;
			float y = 60;
			x *= (numCols / standardLen);
			y *= (numCols / standardLen);

			int thickness = 6 * (numCols / standardLen) + 0.5;

			float scale = numCols / standardLen * 2;
			cv::putText(img, std::string(text), cv::Point2f(numCols - x, y), 0, scale, cv::Scalar(0, 0, 255, 1), thickness, CV_AA);

			std::string writePath = "C:/Users/v-cz/Dropbox/CVPR'15/RenderResults/" + folders[i] + "/" + methodNames[j] + "_withPSNR.png";
			cv::imwrite(writePath, img); 

			std::cout << filePath << "\n";
		}
		
	}


	std::string filePath = "C:/Users/v-cz/Dropbox/CVPR'15/RenderResults/Books/Volume.png";
	*/
}