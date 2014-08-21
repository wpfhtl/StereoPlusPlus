#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ctime>

#include "StereoAPI.h"

#define SIMPROPRADIUS				17
#define COLOR_GAMMA					30


struct SortByLabDist {
	bool operator ()(const std::pair<cv::Point2f, int> &a, const std::pair<cv::Point2f, int> &b) const {
		return a.second < b.second;
	}
};

static cv::Point2i RamdomNbingPosition(int y, int x, int radiusY, int radiusX, int numRows, int numCols)
{
	// Returns a VALID neigboring position of (x, y) in the square spanned by (x, y) and radiusX, radiusY.
	// The function forces (x, y) to be also a valid position, otherwise there may be mod-by-zero trap!!!!
	x = std::max(0, std::min(numCols - 1, x));
	y = std::max(0, std::min(numRows - 1, y));

	int xL = std::max(0, x - radiusX);
	int yU = std::max(0, y - radiusY);
	int xR = std::min(numCols - 1, x + radiusX);
	int yD = std::min(numRows - 1, y + radiusY);
	int xx = xL + rand() % (xR - xL + 1);
	int yy = yU + rand() % (yD - yU + 1);
	return cv::Point2i(xx, yy);
}

void InitSimVecWeights(cv::Mat &img, std::vector<SimVector> &simVecs)
{
	int numRows = img.rows, numCols = img.cols;

	for (int y = 0, id = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++, id++) {
			cv::Vec3b cA = img.at<cv::Vec3b>(y, x);

			for (int i = 0; i < SIMVECTORSIZE; i++) {
				cv::Vec3b cB = img.at<cv::Vec3b>(simVecs[id].pos[i].y, simVecs[id].pos[i].x);
				float diff = L1Dist(cA, cB);
				simVecs[id].w[i] = exp(-diff / COLOR_GAMMA);
			}
		}
	}
}

void SelfSimilarityPropagation(cv::Mat &img, cv::vector<SimVector> &simVecs)
{
	int numRows = img.rows, numCols = img.cols;
	simVecs.resize(numRows * numCols);
	cv::Mat imgLab;
	cv::cvtColor(img, imgLab, CV_BGR2Lab);

	int tic = clock();
	// Step 1 - Init each pixel's similarity vector by random
	for (int y = 0, id = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++, id++) {
			for (int i = 0; i < SIMVECTORSIZE; i++) {
				// We currently does not check for duplicates
				simVecs[id].pos[i] = RamdomNbingPosition(y, x, SIMPROPRADIUS, SIMPROPRADIUS, numRows, numCols);
			}
		}
	}

	// Step 2 - Propagate from upper left to lower right and in reverse order
	int ycInit[2] = { 0, numRows - 1 }, ycEnd[2] = { numRows, -1 };
	int xcInit[2] = { 0, numCols - 1 }, xcEnd[2] = { numCols, -1 };
	int idInit[2] = { 0, numRows * numCols - 1 };

	for (int round = 0; round < 2; round++) {
		int step = (round == 0 ? +1 : -1);

		for (int yc = ycInit[round], id = idInit[round]; yc != ycEnd[round]; yc += step) {
			for (int xc = xcInit[round]; xc != xcEnd[round]; xc += step, id += step) {

				// Merge the similarity vectors from the two neighbors and itself
				//std::vector<cv::Point2i> candidates(simVecs[id].pos, simVecs[id].pos + SIMVECTORSIZE);
				std::vector<cv::Point2i> candidates; candidates.reserve(3 * SIMVECTORSIZE);
				candidates.insert(candidates.end(), simVecs[id].pos, simVecs[id].pos + SIMVECTORSIZE);
				if (0 <= yc - step && yc - step < numRows) {
					int nbId = id - step * numCols;
					candidates.insert(candidates.end(), simVecs[nbId].pos, simVecs[nbId].pos + SIMVECTORSIZE);
				}
				if (0 <= xc - step && xc - step < numCols) {
					int nbId = id - step * 1;
					candidates.insert(candidates.end(), simVecs[nbId].pos, simVecs[nbId].pos + SIMVECTORSIZE);
				}

				// Replace out-of-bound candidates with ramdom valid positions
				for (int i = 0; i < candidates.size(); i++) {
					cv::Point2i diff = candidates[i] - cv::Point2i(xc, yc);
					if (std::abs(diff.x) > SIMPROPRADIUS || std::abs(diff.y) > SIMPROPRADIUS) {
						candidates[i] = RamdomNbingPosition(yc, xc, SIMPROPRADIUS, SIMPROPRADIUS, numRows, numCols);
					}
				}

				// Sort the candidates according to similarity to center pixel
				std::vector<std::pair<cv::Point2i, int>> cddPairs(candidates.size());
				cv::Vec3b cc = imgLab.at<cv::Vec3b>(yc, xc);
				for (int i = 0; i < cddPairs.size(); i++) {
					cddPairs[i].first = candidates[i];
					cv::Vec3b c = imgLab.at<cv::Vec3b>(candidates[i].y, candidates[i].x);
					cddPairs[i].second = L1Dist(c, cc);
				}
				std::sort(cddPairs.begin(), cddPairs.end(), SortByLabDist());

				// Remove duplicates
				std::vector<std::pair<cv::Point2i, int>>::iterator itTail
					= std::unique(cddPairs.begin(), cddPairs.end());
				int numUniqueCandidates = std::distance(cddPairs.begin(), itTail);
				for (int i = 0; i < std::min(numUniqueCandidates, SIMVECTORSIZE); i++) {
					simVecs[id].pos[i] = cddPairs[i].first;
				}
				if (numUniqueCandidates < SIMVECTORSIZE) {
					for (int i = numUniqueCandidates; i < SIMVECTORSIZE; i++) {
						simVecs[id].pos[i] = RamdomNbingPosition(yc, xc, SIMPROPRADIUS, SIMPROPRADIUS, numRows, numCols);
					}
				}
			}
		}
	}

	printf("Self Similarity Propagation costs %.2fs\n", (clock() - tic) / 1000.f);

#if 0
	// Visualize and verify if the algorithm is doing corretly
	cv::Mat nbImg(numRows, numCols, CV_8UC3), canvas;
	nbImg.setTo(cv::Vec3b(0, 0, 0));
	cv::hconcat(nbImg, img, canvas);
	cv::imshow("TestSelfSimilarityPropagation", canvas);
	void* params[2] = { &canvas, &simVecs };
	void OnMouseTestSelfSimilarityPropagation(int event, int x, int y, int flags, void *param);
	cv::setMouseCallback("TestSelfSimilarityPropagation", OnMouseTestSelfSimilarityPropagation, params);
	cv::waitKey(0);
#endif
}