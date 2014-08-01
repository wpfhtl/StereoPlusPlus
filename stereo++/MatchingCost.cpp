#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "MCImg.h"
#include "SlantedPlane.h"
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


#define ASSERT(condition)								\
	if (!condition) {									\
		printf("ASSERT %s VIOLATED AT LINE %d, %s\n",	\
			#condition, __LINE__, __FILE__);			\
		exit(-1);										\
	} 



int		PATCHRADIUS		= 17;
int		PATCHWIDTH		= 35;
float	GRANULARITY		= 0.25f;

enum CostAggregationType	{ REGULAR_GRID, TOP50 };
enum MatchingCostType		{ ADGRADIENT, ADCENSUS };

CostAggregationType		gCostAggregationType	= REGULAR_GRID;
MatchingCostType		gMatchingCostType		= ADCENSUS;

MCImg<float>			gDsiL;
MCImg<float>			gDsiR;
MCImg<float>			gSimWeightsL;
MCImg<float>			gSimWeightsR;
MCImg<SimVector>		gSimVecsL;
MCImg<SimVector>		gSimVecsR;



bool InBound(int y, int x, int numRows, int numCols)
{
	return 0 <= y && y < numRows && 0 <= x && x < numCols;
}

cv::Mat ComputeCensusImage(cv::Mat &img)
{
	// Assumes input is a BGR image or a 3-channel gray image.
	int numRows = img.rows, numCols = img.cols;
	cv::Mat imgGray;
	cv::cvtColor(img, imgGray, CV_BGR2GRAY);

	// We use long long for each element of censusImg, 
	// but CV_64FC1 is the only suitable size we can choose.
	cv::Mat censusImg(numRows, numCols, CV_64FC1);
	ASSERT(censusImg.isContinuous())
		long long *census = (long long*)censusImg.data;

	int vpad = 3;
	int hpad = 4;

	memset(census, 0, imgGray.rows * imgGray.cols * sizeof(long long));
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {

			int uu = std::max(i - vpad, 0);
			int dd = std::min(i + vpad, numRows - 1);
			int ll = std::max(j - hpad, 0);
			int rr = std::min(j + hpad, numCols - 1);

			int idx = 0;
			long long feature = 0;
			uchar center = imgGray.at<unsigned char>(i, j);

			for (int ii = uu; ii <= dd; ii++) {
				for (int jj = ll; jj <= rr; jj++) {
					feature |= ((long long)(imgGray.at<unsigned char>(ii, jj) > center) << idx);
					idx++;
				}
			}

			census[i * numCols + j] = feature;
		}
	}

	return censusImg;
}

cv::Mat ComputeGradientImage(cv::Mat &img)
{
	int numRows = img.rows, numCols = img.cols;
	cv::Mat grayImg;
	cv::Mat gradientImg(numRows, numCols, CV_32FC4);
	cv::Mat grayfImg(numRows + 2, numCols + 2, CV_32FC1);

	cv::cvtColor(img, grayImg, CV_BGR2GRAY);
	grayImg.convertTo(grayImg, CV_32FC1, 1 / 255.f);
	grayImg.copyTo(grayfImg(cv::Rect(1, 1, numCols, numRows)));


	#define BUF(y, x) grayfImg.at<float>(y + 1, x + 1)
	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {

			float dx = (BUF(y + 0, x + 1) - BUF(y + 0, x - 1)) * 0.5
					 + (BUF(y - 1, x + 1) - BUF(y - 1, x - 1)) * 0.25
					 + (BUF(y + 1, x + 1) - BUF(y + 1, x - 1)) * 0.25;

			float dy = (BUF(y + 1, x + 0) - BUF(y - 1, x + 0)) * 0.5
					 + (BUF(y + 1, x - 1) - BUF(y - 1, x - 1)) * 0.25
					 + (BUF(y + 1, x + 1) - BUF(y - 1, x + 1)) * 0.25;

			float dxy = BUF(y + 1, x + 1) - BUF(y - 1, x - 1);
			float dyx = BUF(y + 1, x - 1) - BUF(y - 1, x + 1);

			gradientImg.at<cv::Vec4f>(y, x) = cv::Vec4f(dx, dy, dxy, dyx);
		}
	}

	return gradientImg;
}

int L1Dist(const cv::Vec3b &a, const cv::Vec3b &b)
{
	return std::abs((int)a[0] - (int)b[0])
		 + std::abs((int)a[1] - (int)b[1])
		 + std::abs((int)a[2] - (int)b[2]);
}

float L1Dist(const cv::Vec3f &a, const cv::Vec3f &b)
{
	return std::abs(a[0] - b[0]) 
		 + std::abs(a[1] - b[1]) 
		 + std::abs(a[2] - b[2]);
}

float L1Dist(const cv::Vec4f &a, const cv::Vec4f &b)
{
	return std::abs(a[0] - b[0])
		 + std::abs(a[1] - b[1])
		 + std::abs(a[2] - b[2])
		 + std::abs(a[3] - b[3]);
}

int HammingDist(const long long x, const long long y)
{
	int dist = 0;
	long long val = x ^ y; // XOR
	// Count the number of set bits
	while (val)
	{
		++dist;
		val &= val - 1;
	}
	return dist;
}

MCImg<float> ComputeAdCensusCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity)
{
	#define AD_LAMBDA		30.f
	#define CENSUS_LAMBDA	30.f

	int numRows = imL.rows, numCols = imL.cols;
	int numLevels = numDisps / granularity;
	MCImg<float> dsiL(numRows, numCols, numLevels);

	cv::Mat censusImgL = ComputeCensusImage(imL);
	cv::Mat censusImgR = ComputeCensusImage(imR);
	cv::Mat bgrL;	imL.convertTo(bgrL, CV_32FC3, 1.f);
	cv::Mat bgrR;	imR.convertTo(bgrR, CV_32FC3, 1.f);

	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			for (int level = 0; level < numLevels; level++) {

				float d = level * granularity;
				float xm = x + sign * d;

				// FIXME: has implicitly assumed "ndisps <= numCols", it's not safe.
				if (xm < 0)			xm += numCols;
				if (xm > numCols - 1) xm -= numCols;

				float xmL, xmR, wL, wR;
				float cost_col, cost_grad;
				xmL = (int)(xm);
				xmR = (int)(xm + 0.99);
				wL = xmR - xm;
				wR = 1.f - wL;

				cv::Vec3f &colorL	= bgrL.at<cv::Vec3f>(y, x);
				cv::Vec3f &colorRmL = bgrR.at<cv::Vec3f>(y, xmL);
				cv::Vec3f &colorRmR = bgrR.at<cv::Vec3f>(y, xmR);
				cv::Vec3f colorR	= wL * colorRmL + wR * colorRmR;
				long long censusL	= censusImgL.at<long long>(y, x);
				long long censusR	= censusImgR.at<long long>(y, xm + 0.5);

				float adDiff = 0.3333f * L1Dist(colorL, colorR);
				float censusDiff = HammingDist(censusL, censusR);
				dsiL.get(y, x)[level] = 2.f - exp(-adDiff / AD_LAMBDA) - exp(-censusDiff / CENSUS_LAMBDA);
			}
		}
	}

	return dsiL;


}

MCImg<float> ComputeAdGradientCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity)
{
	#define COLORGRADALPHA	0.05f
	#define COLORMAXDIFF	0.04f
	#define GRADMAXDIFF		0.01f

	int numRows = imL.rows, numCols = imL.cols;
	int numLevels = numDisps / granularity;
	MCImg<float> dsiL(numRows, numCols, numLevels);

	cv::Mat gradientL = ComputeGradientImage(imL);
	cv::Mat gradientR = ComputeGradientImage(imR);
	cv::Mat bgrL;	imL.convertTo(bgrL, CV_32FC3, 1 / 255.f);
	cv::Mat bgrR;	imR.convertTo(bgrR, CV_32FC3, 1 / 255.f);

	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			for (int level = 0; level < numLevels; level++) {

				float d = level * granularity;
				float xm = x + sign * d;

				// FIXME: has implicitly assumed "ndisps <= numCols", it's not safe.
				if (xm < 0)				xm += numCols;
				if (xm > numCols - 1)	xm -= numCols;

				float xmL, xmR, wL, wR;
				xmL = (int)(xm);
				xmR = (int)(xm + 0.99);
				wL = xmR - xm;
				wR = 1.f - wL;

				cv::Vec3f &colorL	= bgrL.at<cv::Vec3f>(y, x);
				cv::Vec3f &colorRmL	= bgrR.at<cv::Vec3f>(y, xmL);
				cv::Vec3f &colorRmR	= bgrR.at<cv::Vec3f>(y, xmR);
				cv::Vec4f &gradL	= gradientL.at<cv::Vec4f>(y, x);
				cv::Vec4f &gradRmL	= gradientR.at<cv::Vec4f>(y, xmL);
				cv::Vec4f &gradRmR	= gradientR.at<cv::Vec4f>(y, xmR);
				cv::Vec3f colorR	= wL * colorRmL + wR * colorRmR;
				cv::Vec4f gradR		= wL * gradRmL + wR * gradRmR;

				float costColor = std::min(COLORMAXDIFF, L1Dist(colorL, colorR));
				float costGrad  = std::min(GRADMAXDIFF,  L1Dist(gradL, gradR));
				dsiL.get(y, x)[level] = COLORGRADALPHA * costColor + (1 - COLORGRADALPHA) * costGrad;
			}
		}
	}

	return dsiL;
}

cv::Mat WinnerTakesAll(MCImg<float> &dsi, float granularity)
{
	int numRows = dsi.h, numCols = dsi.w, ndisps = dsi.n;
	cv::Mat disp(numRows, numCols, CV_32FC1);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int minidx = 0;
			float *cost = dsi.get(y, x);
			for (int k = 1; k < ndisps; k++) {
				if (cost[k] < cost[minidx]) {
					minidx = k;
				}
			}
			disp.at<float>(y, x) = minidx * granularity;
		}
	}
	return disp;
}

float PatchMatchSlantedPlaneCost(int yc, int xc, SlantedPlane &slantedPlane, int sign)
{
	MCImg<float> &dsi = (sign == -1 ? gDsiL : gDsiR);
	MCImg<float> &simWeights = (sign == -1 ? gSimWeightsL : gSimWeightsR);
	MCImg<SimVector> &simVecs = (sign == -1 ? gSimVecsL : gSimVecsR);

	int numRows = dsi.h, numCols = dsi.w, maxLevel = dsi.n - 1;
	const int STRIDE = 1;
	float totalCost = 0.f;

	if (gCostAggregationType == REGULAR_GRID) {
		MCImg<float> w(PATCHWIDTH, PATCHWIDTH, 1, simWeights.line(yc * numCols + xc));
		for (int y = yc - PATCHRADIUS, id = 0; y <= yc + PATCHRADIUS; y += STRIDE) {
			for (int x = xc - PATCHRADIUS; x <= xc + PATCHRADIUS; x += STRIDE, id++) {
				if (InBound(y, x, numRows, numCols)) {
					id = (y - (yc - PATCHRADIUS)) * PATCHWIDTH + (x - (xc - PATCHRADIUS));
					float d = slantedPlane.ToDisparity(y, x);
					int level = 0.5 + d / GRANULARITY;
					level = std::max(0, std::min(maxLevel, level));
					totalCost += w.data[id] * dsi.get(y, x)[level];
				}
			}
		}
	}
	else if (gCostAggregationType == TOP50) {
		SimVector &simVec = simVecs[yc][xc];
		for (int i = 0; i < SIMVECTORSIZE; i++) {
			int y = simVec.pos[i].y;
			int x = simVec.pos[i].x;
			float d = slantedPlane.ToDisparity(y, x);
			int level = 0.5 + d / GRANULARITY;
			level = std::max(0, std::min(maxLevel, level));
			totalCost += simVec.w[i] * dsi.get(y, x)[level];
		}
	}

	return totalCost;
}

