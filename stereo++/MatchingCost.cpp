#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "MCImg.h"
#include "SlantedPlane.h"
#include "StereoAPI.h"
#include "ReleaseAssert.h"
#include "Timer.h"






extern int				PATCHRADIUS;
extern int				PATCHWIDTH;
extern float			GRANULARITY;

extern enum CostAggregationType		{ GRID, TOP50 };
extern enum MatchingCostType		{ ADGRADIENT, ADCENSUS };
extern CostAggregationType			gCostAggregationType;
extern MatchingCostType				gMatchingCostType;

extern MCImg<float>			gDsiL;
extern MCImg<float>			gDsiR;
extern MCImg<float>			gSimWeightsL;
extern MCImg<float>			gSimWeightsR;
extern MCImg<SimVector>		gSimVecsL;
extern MCImg<SimVector>		gSimVecsR;



bool InBound(int y, int x, int numRows, int numCols)
{
	return 0 <= y && y < numRows && 0 <= x && x < numCols;
}

bool InBound(cv::Point &p, int numRows, int numCols)
{
	return 0 <= p.y && p.y < numRows && 0 <= p.x && p.x < numCols;
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
	grayfImg.setTo(0.f);

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

cv::Mat ComputeGradxyImage(cv::Mat& img)
{
	int numRows = img.rows, numCols = img.cols;
	int sobelScale = 1, sobelDelta = 0;
	cv::Mat gray, grad_x, grad_y;

	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3, sobelScale, sobelDelta, cv::BORDER_DEFAULT);
	cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3, sobelScale, sobelDelta, cv::BORDER_DEFAULT);
	grad_x = grad_x / 8.f;
	grad_y = grad_y / 8.f;

	cv::Mat grad(numRows, numCols, CV_32FC2);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			grad.at<cv::Vec2f>(y, x)[0] = grad_x.at<float>(y, x);
			grad.at<cv::Vec2f>(y, x)[1] = grad_y.at<float>(y, x);
		}
	}

	return grad / 255.f;
}

int L1Dist(const cv::Vec3b &a, const cv::Vec3b &b)
{
	return std::abs((int)a[0] - (int)b[0])
		 + std::abs((int)a[1] - (int)b[1])
		 + std::abs((int)a[2] - (int)b[2]);
}

float L1Dist(const cv::Vec2f &a, const cv::Vec2f &b)
{
	return std::abs(a[0] - b[0]) + std::abs(a[1] - b[1]);
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

#if 1
MCImg<float> ComputeAdGradientCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity)
{
	//#define COLORGRADALPHA	0.05f
	//#define COLORMAXDIFF		0.04f
	//#define GRADMAXDIFF		0.01f

	extern float COLORGRADALPHA;
	extern float COLORMAXDIFF;
	extern float GRADMAXDIFF;

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
#else
MCImg<float> ComputeAdGradientCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign, float granularity)
{
	const float	ALPHA		= 0.1;
	const float	TAU_COLOR	= 10;
	const float	TAU_GRAD	= 2;

	extern float COLORGRADALPHA;
	extern float COLORMAXDIFF;
	extern float GRADMAXDIFF;

	int numRows = imL.rows, numCols = imL.cols;
	int numLevels = numDisps / granularity;
	MCImg<float> dsiL(numRows, numCols, numLevels);

	cv::Mat gradientL = ComputeGradxyImage(imL);
	cv::Mat gradientR = ComputeGradxyImage(imR);
	cv::Mat bgrL;	imL.convertTo(bgrL, CV_32FC3, 1.f / 255);
	cv::Mat bgrR;	imR.convertTo(bgrR, CV_32FC3, 1.f / 255);

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
				cv::Vec3f &colorRmL = bgrR.at<cv::Vec3f>(y, xmL);
				cv::Vec3f &colorRmR = bgrR.at<cv::Vec3f>(y, xmR);
				cv::Vec2f &gradL	= gradientL.at<cv::Vec2f>(y, x);
				cv::Vec2f &gradRmL	= gradientR.at<cv::Vec2f>(y, xmL);
				cv::Vec2f &gradRmR	= gradientR.at<cv::Vec2f>(y, xmR);
				cv::Vec3f colorR	= wL * colorRmL + wR * colorRmR;
				cv::Vec2f gradR		= wL * gradRmL  + wR * gradRmR;


				//float costColor = std::min(TAU_COLOR, L1Dist(colorL, colorR));
				//float costGrad  = std::min(TAU_GRAD,  L1Dist(gradL, gradR));
				//dsiL.get(y, x)[level] = ALPHA * costColor + (1 - ALPHA) * costGrad;

				float costColor = std::min(COLORMAXDIFF, L1Dist(colorL, colorR));
				float costGrad  = std::min(GRADMAXDIFF, L1Dist(gradL, gradR));
				dsiL.get(y, x)[level] = COLORGRADALPHA * costColor + (1 - COLORGRADALPHA) * costGrad;
			}
		}
	}

	return dsiL;
}
#endif

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

template<typename T> static inline T BilinearInterp(cv::Mat &im, int y, float x)
{
	int numRows = im.rows, numCols = im.cols;
	ASSERT(0 <= y && y < numRows);
	
	x = std::max(0.f, std::min(numCols - 1.f, x));
	int xL = floor(x);
	int xR = ceil(x);
	float wL = std::abs(x - xR);
	float wR = 1.f - wL;
	return wL * im.at<T>(y, xL) + wR * im.at<T>(y, xR);
}

float PatchMatchSlantedPlaneCost(int yc, int xc, SlantedPlane &slantedPlane, int sign)
{
	extern int INTERP_ONLINE;

	if (INTERP_ONLINE) {
		extern float SIMILARITY_GAMMA;
		extern cv::Mat gImRgbL, gImRgbR, gImGradL, gImGradR;
		int numRows = gImRgbL.rows, numCols = gImRgbL.cols;
		extern float COLORGRADALPHA;
		extern float COLORMAXDIFF;
		extern float GRADMAXDIFF;

		// The color and gradient feature are in range [0, 1]
		cv::Mat &imRgbL		= (sign == -1 ? gImRgbL : gImRgbR);
		cv::Mat &imRgbR		= (sign == -1 ? gImRgbR : gImRgbL);
		cv::Mat &imGradL	= (sign == -1 ? gImGradL : gImGradR);
		cv::Mat &imGradR	= (sign == -1 ? gImGradR : gImGradL);

		cv::Vec3f c = imRgbL.at<cv::Vec3f>(yc, xc);
		float wsum = 0.f;
		float totalCost = 0.f;

		for (int y = yc - PATCHRADIUS, id = 0; y <= yc + PATCHRADIUS; y += 1) {
			for (int x = xc - PATCHRADIUS; x <= xc + PATCHRADIUS; x += 1, id++) {
				if (InBound(y, x, numRows, numCols)) {

					float w = exp(-255.f * L1Dist(c, imRgbL.at<cv::Vec3f>(y, x)) / SIMILARITY_GAMMA);
					float d = slantedPlane.ToDisparity(y, x);
					float xm = x + sign * d;

					cv::Vec3f colorL = imRgbL.at<cv::Vec3f>(y, x);
					cv::Vec3f colorR = BilinearInterp<cv::Vec3f>(imRgbR, y, xm);
					cv::Vec4f gradL  = imGradL.at<cv::Vec4f>(y, x);
					cv::Vec4f gradR  = BilinearInterp<cv::Vec4f>(imGradR, y, xm);
					
					float costColor = std::min(COLORMAXDIFF, L1Dist(colorL, colorR));
					float costGrad  = std::min(GRADMAXDIFF,  L1Dist(gradL, gradR));
					float cost = COLORGRADALPHA * costColor + (1 - COLORGRADALPHA) * costGrad;

					totalCost += w * cost;
					wsum += w;
				}
			}
		}

		if (wsum <= 1.1f) {
			return 1e5f;
		}
		return totalCost / wsum;
	}



	MCImg<float> &dsi			= (sign == -1 ? gDsiL : gDsiR);
	MCImg<float> &simWeights	= (sign == -1 ? gSimWeightsL : gSimWeightsR);
	MCImg<SimVector> &simVecs	= (sign == -1 ? gSimVecsL : gSimVecsR);

	int numRows = dsi.h, numCols = dsi.w, maxLevel = dsi.n - 1;
	const int STRIDE = 1;
	float totalCost = 0.f;
	float accWeight = 0.f;

	if (gCostAggregationType == GRID) {
		MCImg<float> w(PATCHWIDTH, PATCHWIDTH, 1, simWeights.line(yc * numCols + xc));
		for (int y = yc - PATCHRADIUS, id = 0; y <= yc + PATCHRADIUS; y += STRIDE) {
			for (int x = xc - PATCHRADIUS; x <= xc + PATCHRADIUS; x += STRIDE, id++) {
				if (InBound(y, x, numRows, numCols)) {
					id = (y - (yc - PATCHRADIUS)) * PATCHWIDTH + (x - (xc - PATCHRADIUS));
					float d = slantedPlane.ToDisparity(y, x);
					int level = 0.5 + d / GRANULARITY;
					level = std::max(0, std::min(maxLevel, level));
					totalCost += w.data[id] * dsi.get(y, x)[level];
					accWeight += w.data[id];
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
			accWeight += simVec.w[i];
		}
	}

	//return totalCost / numPixelsInBound;
	return totalCost / accWeight;
}

void InitGlobalColorGradientFeatures(cv::Mat &imL, cv::Mat &imR)
{
	extern cv::Mat gImRgbL, gImRgbR, gImGradL, gImGradR;
	imL.convertTo(gImRgbL, CV_32FC3, 1.f / 255.f);
	imR.convertTo(gImRgbR, CV_32FC3, 1.f / 255.f);
	gImGradL = ComputeGradientImage(imL);
	gImGradR = ComputeGradientImage(imR);
}

void InitGlobalDsiAndSimWeights(cv::Mat &imL, cv::Mat &imR, int numDisps)
{
	extern float SIMILARITY_GAMMA;
	int numRows = imL.rows, numCols = imL.cols;
	if (gMatchingCostType == ADCENSUS) {
		gDsiL = ComputeAdCensusCostVolume(imL, imR, numDisps, -1, GRANULARITY);
		gDsiR = ComputeAdCensusCostVolume(imR, imL, numDisps, +1, GRANULARITY);
	}
	else if (gMatchingCostType == ADGRADIENT) {
		gDsiL = ComputeAdGradientCostVolume(imL, imR, numDisps, -1, GRANULARITY);
		gDsiR = ComputeAdGradientCostVolume(imR, imL, numDisps, +1, GRANULARITY);
	}

	std::vector<SimVector> simVecsStdL;
	std::vector<SimVector> simVecsStdR;

	if (gCostAggregationType == GRID) {
		bs::Timer::Tic("Precompute Similarity Weights");
		MCImg<float> PrecomputeSimilarityWeights(cv::Mat &img, int patchRadius, int simGamma);
		gSimWeightsL = PrecomputeSimilarityWeights(imL, PATCHRADIUS, SIMILARITY_GAMMA);
		gSimWeightsR = PrecomputeSimilarityWeights(imR, PATCHRADIUS, SIMILARITY_GAMMA);
		bs::Timer::Toc();
	}
	else if (gCostAggregationType == TOP50) {
		bs::Timer::Tic("Begin SelfSimilarityPropagation");
		SelfSimilarityPropagation(imL, simVecsStdL);
		InitSimVecWeights(imL, simVecsStdL);
		gSimVecsL = MCImg<SimVector>(numRows, numCols, 1, &simVecsStdL[0]);
		SelfSimilarityPropagation(imR, simVecsStdR);
		InitSimVecWeights(imR, simVecsStdR);
		gSimVecsR = MCImg<SimVector>(numRows, numCols, 1, &simVecsStdR[0]);
		bs::Timer::Toc();
	}
}