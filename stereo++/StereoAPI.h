#include <opencv2/core/core.hpp>
#include "MCImg.h"

#define SIMVECTORSIZE	50

struct SimVector
{
	cv::Point2i pos[SIMVECTORSIZE];
	float w[SIMVECTORSIZE];
};

cv::Mat ComputeCensusImage(cv::Mat &img);

cv::Mat ComputeGradientImage(cv::Mat &img);

int L1Dist(const cv::Vec3b &a, const cv::Vec3b &b);

float L1Dist(const cv::Vec3f &a, const cv::Vec3f &b);

float L1Dist(const cv::Vec4f &a, const cv::Vec4f &b);

int HammingDist(const long long x, const long long y);

MCImg<float> ComputeAdCensusCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity);

MCImg<float> ComputeAdGradientCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity);

cv::Mat WinnerTakesAll(MCImg<float> &dsi, float granularity);

void SaveDisparityToPly(cv::Mat &disp, cv::Mat& img, float maxDisp,
	std::string workingDir, std::string plyFilePath, cv::Mat &validPixelMap = cv::Mat());

void SetupStereoParameters(std::string rootFolder, int &numDisps, int &maxDisp, int &visualizeScale);

void EvaluateDisparity(std::string rootFolder, cv::Mat &dispL, float eps = 1.f);

void RunPatchMatchOnPixels(std::string rootFolder, cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR);

void InitSimVecWeights(cv::Mat &img, std::vector<SimVector> &simVecs);

void SelfSimilarityPropagation(cv::Mat &img, cv::vector<SimVector> &simVecs);

void Triangulate2DImage(cv::Mat& img, std::vector<cv::Point2d> &vertexCoords, std::vector<std::vector<int>> &triVertexInds);