#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <stack>
#include <string>

#include "StereoAPI.h"
#include "SlantedPlane.h"



static int SegmentDisparityMap(cv::Mat &dispGT, cv::Mat &segMap, int thresh)
{
	// The flood fill segmentation not effective. 
	// Many regions of different planes are grouped to the same segment.

	const cv::Point2i dirDelta[4] = { cv::Point2i(-1, 0), cv::Point2i(+1, 0), cv::Point2i(0, -1), cv::Point2i(0, +1) };

	int numSegs = 0;
	int numRows = dispGT.rows, numCols = dispGT.cols;
	segMap.create(numRows, numCols, CV_32SC1);
	segMap.setTo(-1);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			if (segMap.at<int>(y, x) == -1 && dispGT.at<float>(y, x) != 0.f) {
				segMap.at<int>(y, x) = numSegs;
				std::stack<cv::Point2i> stack;
				stack.push(cv::Point2i(x, y));
				
				while (!stack.empty()) {
					cv::Point2i p = stack.top();
					stack.pop();
					float dp = dispGT.at<float>(p.y, p.x);

					for (int k = 0; k < 4; k++) {
						cv::Point2i q = p + dirDelta[k];
						if (InBound(q, numRows, numCols) && segMap.at<int>(q.y, q.x) == -1) {
							float dq = dispGT.at<float>(q.y, q.x);
							if (dq != 0.f && std::abs(dp - dq) <= thresh) {
								segMap.at<int>(q.y, q.x) = numSegs;
								stack.push(cv::Point2i(q));
							}
						}
					}
				}
				numSegs++;
			}
		}
	}

	return numSegs;
}

static SlantedPlane PlaneFit(std::vector<cv::Point3d> &xyd)
{
	int n = xyd.size();
	cv::Mat A(n, 3, CV_32FC1), B(n, 1, CV_32FC1), X;
	for (int i = 0; i < n; i++) {
		A.at<float>(i, 0) = xyd[i].x;
		A.at<float>(i, 1) = xyd[i].y;
		A.at<float>(i, 2) = 1.f;
		B.at<float>(i, 0) = xyd[i].z;
	}
	cv::solve(A, B, X, cv::DECOMP_QR);
	float a = X.at<float>(0, 0);
	float b = X.at<float>(1, 0);
	float c = X.at<float>(2, 0);
	return SlantedPlane::ConstructFromAbc(a, b, c);
}

static void ExtractGroundTruthPlanes(std::string rootFolder)
{
	//int n = 100;
	//cv::Mat A(n, 3, CV_32FC1), B(n, 1, CV_32FC1), X;
	//for (int i = 0; i < n; i++) {
	//	A.at<float>(i, 0) = (float)rand() / INT_MAX;
	//	A.at<float>(i, 1) = (float)rand() / INT_MAX;
	//	A.at<float>(i, 2) = (float)rand() / INT_MAX;
	//	B.at<float>(i, 0) = (float)rand() / INT_MAX;
	//}
	//cv::solve(A, B, X, cv::DECOMP_QR);
	//printf("fsfsfadfasdf\n");
	//return;



	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);

	cv::Mat imL		= cv::imread("d:/data/stereo/" + rootFolder + "/im2.png");
	cv::Mat dispImg = cv::imread("d:/data/stereo/" + rootFolder + "/disp2.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat dispGT;
	dispImg.convertTo(dispGT, CV_32FC1, 1.f / visualizeScale);

	// Flood Fill Segmentation
	cv::Mat segMap, contourImg;
	//int numSegs = SegmentDisparityMap(dispGT, segMap, 0.5f);
	int SLICSegmentation(const cv::Mat &img, const int numPreferedRegions, const int compactness, cv::Mat& labelMap, cv::Mat& contourImg);
	int numSegs = SLICSegmentation(imL, 400, 20, segMap, contourImg);

	// Plane Fitting
	std::vector<std::vector<cv::Point3d>> xyds(numSegs);
	std::vector<std::vector<cv::Point2i>> regioniPixelLists(numSegs);
	int numRows = dispGT.rows, numCols = dispGT.cols;
	for (int y = 0; y < numRows; y++){
		for (int x = 0; x < numCols; x++) {
			int id = segMap.at<int>(y, x);
			float d = dispGT.at<float>(y, x);
			if (id >= 0) {
				xyds[id].push_back(cv::Point3d(x, y, d));
				regioniPixelLists[id].push_back(cv::Point2i(x, y));
			}
		}
	}

	std::vector<SlantedPlane> gtPlanes(numSegs);
	for (int id = 0; id < numSegs; id++) {
		if (xyds[id].size() > 10) {
			gtPlanes[id] = PlaneFit(xyds[id]);
		}
		else {
			gtPlanes[id] = SlantedPlane::ConstructFromAbc(0.f, 0.f, 30.f);
		}
	}
	
	// Statistics
	float minNz = FLT_MAX, maxNz = -FLT_MAX;
	for (int id = 0; id < numSegs; id++) {
		SlantedPlane p = gtPlanes[id];
		//printf("( a,  b,  c) = %(%f, %f, %f)\n", p.a,  p.b,  p.c);
		//printf("(nx, ny, nz) = %(%f, %f, %f)\n", p.nx, p.ny, p.nz);
		minNz = std::min(minNz, p.nz);
		maxNz = std::max(maxNz, p.nz);
	}
	printf("minNz = %f\n", minNz);
	printf("maxNz = %f\n", maxNz);

	// Colorize segments
	std::vector<cv::Vec3b> colors(numSegs);
	for (int id = 0; id < numSegs; id++) {
		colors[id][0] = rand() % 256;
		colors[id][1] = rand() % 256;
		colors[id][2] = rand() % 256;
	}

	cv::Mat segImg(numRows, numCols, CV_8UC3);
	segImg.setTo(cv::Vec3b(0, 0, 0));
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int id = segMap.at<int>(y, x);
			if (id > -1) {
				segImg.at<cv::Vec3b>(y, x) = colors[id];
			}
		}
	}

#if 1
	cv::Mat canvasL(numRows, numCols, CV_8UC3);  canvasL.setTo(cv::Vec3b(0, 0, 0));
	cv::Mat canvasR(numRows, numCols, CV_8UC3);  canvasR.setTo(cv::Vec3b(0, 0, 0));
	for (int id = 0; id < numSegs; id++) {
		if (gtPlanes[id].nz < 0.7) {
			std::vector<cv::Point2i> &pixelList = regioniPixelLists[id];
			for (int k = 0; k < pixelList.size(); k++) {
				int y = pixelList[k].y;
				int x = pixelList[k].x;
				unsigned char d = visualizeScale * gtPlanes[id].ToDisparity(y, x);

				canvasL.at<cv::Vec3b>(y, x) = cv::Vec3b(d, d, d);
				canvasR.at<cv::Vec3b>(y, x) = colors[id];
			}
		}
	}

	struct GTStatParams {
		cv::Mat *labelMap;
		std::vector<SlantedPlane> *gtPlanes;
	} gtStatParams;
	gtStatParams.labelMap = &segMap;
	gtStatParams.gtPlanes = &gtPlanes;

	cv::Mat canvas;
	cv::hconcat(canvasL, canvasR, canvas);
	cv::cvtColor(dispImg, dispImg, CV_GRAY2BGR);
	cv::hconcat(dispImg, canvas, canvas);
	cv::hconcat(canvas, segImg, canvas);
	cv::imshow("Wrong Planes", canvas);
	void OnMouseGroundTruthPlaneStatistics(int event, int x, int y, int flags, void *param);
	cv::setMouseCallback("Wrong Planes", OnMouseGroundTruthPlaneStatistics, &gtStatParams);
	cv::waitKey(0);
#else 
	struct GTStatParams {
		cv::Mat *labelMap;
		std::vector<SlantedPlane> *gtPlanes;
	} gtStatParams;
	gtStatParams.labelMap = &segMap;
	gtStatParams.gtPlanes = &gtPlanes;

	cv::imshow("segImg", segImg);
	void OnMouseGroundTruthPlaneStatistics(int event, int x, int y, int flags, void *param);
	cv::setMouseCallback("segImg", OnMouseGroundTruthPlaneStatistics, &gtStatParams);
	cv::waitKey(0);
#endif
}


void TestGroundTruthPlaneStatistics()
{
	FILE *fid = fopen("d:/data/stereo/testCaseNames.txt", "r");
	char rootFolder[1024];
	while (fscanf(fid, "%s", rootFolder) != EOF) {
		printf("\ndoing %s ...\n", rootFolder);
		ExtractGroundTruthPlanes(rootFolder);
	}
	fclose(fid);
}