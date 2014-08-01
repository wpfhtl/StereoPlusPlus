#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "StereoAPI.h"

/* Note: each OnMouse callback function should take a fixed list of parameters.
 * It is the caller's responsibility to prepare different parameter lists for
 * different OnMouse fucntions.
 */


void OnMouseEvaluateDisprity(int event, int x, int y, int flags, void *param)
{
	cv::Mat &canvas	= *(cv::Mat*)((void**)param)[0];
	cv::Mat &dispL	= *(cv::Mat*)((void**)param)[1];
	cv::Mat &GT		= *(cv::Mat*)((void**)param)[2];
	cv::Mat &imL	= *(cv::Mat*)((void**)param)[3];
	cv::Mat &imR	= *(cv::Mat*)((void**)param)[4];

	float badOnNonocc	= *(float*)((void**)param)[5];
	float badOnAll		= *(float*)((void**)param)[6];
	float badOnDisc		= *(float*)((void**)param)[7];

	int numDisps		= *(int*)((void**)param)[8];
	int visualizeScale	= *(int*)((void**)param)[9];
	int maxDisp			= numDisps - 1;

	std::string workingDir = *(std::string*)((void**)param)[10];


	int numRows = GT.rows, numCols = GT.cols;
	int originX = x;
	int originY = y;
	x %= numCols;
	y %= numRows;
	cv::Mat tmp = canvas.clone();

	if (event == CV_EVENT_MOUSEMOVE)
	{
		cv::Point CC(x + 1 * numCols, originY);
		cv::Point LL(x + 0 * numCols, originY);
		cv::Point RR(x + 2 * numCols, originY);

		cv::line(tmp, CC, LL, cv::Scalar(255, 0, 0));
		cv::line(tmp, CC, RR, cv::Scalar(255, 0, 0));
		cv::circle(tmp, CC, 1, cv::Scalar(0, 0, 255), 2, CV_AA);

		float dGT = GT.at<float>(y, x);
		float dMY = dispL.at<float>(y, x);
		char text[1024]; 
		sprintf(text, "(%d, %d)  GT: %.2f  MINE: %.2f", y, x, dGT, dMY);
		cv::putText(tmp, std::string(text), cv::Point2d(20, 50), 0, 0.6, cv::Scalar(0, 0, 255, 1), 2);

		cv::Point UU(x + 2 * numCols, y);
		cv::Point DD(x + 2 * numCols - dMY, y + numRows);
		cv::line(tmp, UU, DD, cv::Scalar(0, 0, 255), 1, CV_AA);
	}

	if (event == CV_EVENT_RBUTTONDOWN)
	{
		char text[1024];
		sprintf(text, "BadPixelRate  %.2f%%  %.2f%%  %.2f%%", 100.f * badOnNonocc, 100.f * badOnAll, 100.f * badOnDisc);
		cv::putText(tmp, std::string(text), cv::Point2d(20, 50), 0, 0.6, cv::Scalar(0, 0, 255, 1), 2);
	}

	if (event == CV_EVENT_LBUTTONDBLCLK)
	{
		std::string plyFilePath = workingDir + "/OnMouseEvaluateDisprity.ply";
		std::string cmdInvokeMeshlab = "meshlab " + plyFilePath;
		void SaveDisparityToPly(cv::Mat &disp, cv::Mat& img, float maxDisp,
			std::string workingDir, std::string plyFilePath, cv::Mat &validPixelMap = cv::Mat());
		SaveDisparityToPly(dispL, imL, maxDisp, workingDir, plyFilePath);
		system(cmdInvokeMeshlab.c_str());
	}

	cv::imshow("OnMouseEvaluateDisprity", tmp);
}

void OnMouseTestSelfSimilarityPropagation(int event, int x, int y, int flags, void *param)
{
	cv::Mat canvas = *(cv::Mat*)((void**)param)[0];
	cv::Mat tmp = canvas.clone();

	int nrows = canvas.rows, ncols = canvas.cols / 2;
	x %= ncols;
	y %= nrows;

	if (event == CV_EVENT_MOUSEMOVE)
	{
		cv::Point AA(x, y), BB(x + ncols, y);
		cv::line(tmp, AA, BB, cv::Scalar(255, 0, 0, 1));
	}

	if (event == CV_EVENT_LBUTTONDOWN) {
		std::vector<SimVector> &simVecs = *(std::vector<SimVector>*)((void**)param)[1];
		cv::Mat nbImg = tmp(cv::Rect(0, 0, ncols, nrows));
		cv::Mat img = tmp(cv::Rect(ncols, 0, ncols, nrows));
		nbImg.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
		int id = y * ncols + x;
		for (int i = 0; i < SIMVECTORSIZE; i++) {
			cv::Point2i nb = simVecs[id].pos[i];
			nbImg.at<cv::Vec3b>(nb.y, nb.x) = cv::Vec3b(255, 255, 255);
			img.at<cv::Vec3b>(nb.y, nb.x) = cv::Vec3b(255, 255, 255);
		}
	}

	cv::imshow("TestSelfSimilarityPropagation", tmp);
}