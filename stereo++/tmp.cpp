#include <vector>
#include <algorithm>

#include "MCImg.h"
#include "SlantedPlane.h"


const int patch_r	= 17;
const int patch_w	= 35;
const int nrows		= 375;
const int ncols		= 450;

static void PlaneMapToDisparityMap(MCImg<SlantedPlane>& coeffs, MCImg<float>& disp)
{
	for (int y = 0; y < nrows; y++) {
		for (int x = 0; x < ncols; x++) {
			disp[y][x] = coeffs[y][x].ToDisparity(y, x);
		}
	}
}

static void CrossCheck(MCImg<float>& dispL, MCImg<float>& dispR, MCImg<bool>& validL, MCImg<bool>& validR)
{
	for (int y = 0; y < nrows; y++) {
		for (int x = 0; x < ncols; x++) {

			int xR = std::max(0.f, std::min((float)ncols, x - dispL[y][x]));
			validL[y][x] = (std::abs(dispL[y][x] - dispR[y][xR]) <= 1);

			int xL = std::max(0.f, std::min((float)ncols, x + dispR[y][x]));
			validR[y][x] = (std::abs(dispR[y][x] - dispL[y][xL]) <= 1);
		}
	}
}

static void FillHole(int y, int x, MCImg<bool>& valid, MCImg<SlantedPlane>& coeffs)
{
	// This function fills the invalid pixel (y,x) by finding its nearst (left and right) 
	// valid neighbors on the same scanline, and select the one with lower disparity.

	int xL = x - 1, xR = x + 1, bestx = x;
	while (!valid[y][xL] && 0 <= xL) {
		xL--;
	}
	while (!valid[y][xR] && xR < ncols) {
		xR++;
	}
	if (0 <= xL) {
		bestx = xL;
	}
	if (xR < ncols) {
		if (bestx == xL) {
			float dL = coeffs[y][xL].ToDisparity(y, x);
			float dR = coeffs[y][xR].ToDisparity(y, x);
			if (dR < dL) {
				bestx = xR;
			}
		}
		else {
			bestx = xR;
		}
	}
	coeffs[y][x] = coeffs[y][bestx];
}

void WeightedMedianFilter(int yc, int xc, MCImg<float>& disp, MCImg<float>& weights, MCImg<bool>& valid, bool useInvalidPixels)
{
	std::vector<std::pair<float, float>> dw_pairs;

	int yb = std::max(0, yc - patch_r), ye = std::min(nrows - 1, yc + patch_r);
	int xb = std::max(0, xc - patch_r), xe = std::min(ncols - 1, xc + patch_r);

	for (int y = yb; y <= ye; y++) {
		for (int x = xb; x <= xe; x++) {
			if (useInvalidPixels || valid[y][x]) {
				std::pair<float, float> dw(disp[y][x], weights[y - yc + patch_r][x - xc + patch_r]);
				dw_pairs.push_back(dw);
			}
		}
	}

	std::sort(dw_pairs.begin(), dw_pairs.end());

	float w = 0.f, wsum = 0.f;
	for (int i = 0; i < dw_pairs.size(); i++) {
		wsum += dw_pairs[i].second;
	}

	for (int i = 0; i < dw_pairs.size(); i++) {
		w += dw_pairs[i].second;
		if (w >= wsum / 2.f) {
			// Note that this line can always be reached.
			if (i > 0) {
				disp[yc][xc] = (dw_pairs[i - 1].first + dw_pairs[i].first) / 2.f;
			}
			else {
				disp[yc][xc] = dw_pairs[i].first;
			}
			break;
		}
	}
}

void PostProcess(
	MCImg<float>& weightsL, MCImg<float>& weightsR,
	MCImg<SlantedPlane>& coeffsL, MCImg<SlantedPlane>& coeffsR,
	MCImg<float>& dispL, MCImg<float>& dispR)
{
	// This function perform several iterations of weighted median filtering at each invalid position.
	// weights of the neighborhood are set by exp(-||cp-cq|| / gamma), except in the last iteration,
	// where the weights of invalid pixels are set to zero.

	MCImg<bool> validL(nrows, ncols), validR(nrows, ncols);
//	PlaneMapToDisparityMap(coeffsL, dispL);
//	PlaneMapToDisparityMap(coeffsR, dispR);
//
//	// Hole filling
//	CrossCheck(dispL, dispR, validL, validR);
//#pragma omp parallel for
//	for (int y = 0; y < nrows; y++) {
//		for (int x = 0; x < ncols; x++) {
//			if (!validL[y][x]) {
//				FillHole(y, x, validL, coeffsL);
//			}
//			if (!validR[y][x]) {
//				FillHole(y, x, validR, coeffsR);
//			}
//		}
//	}
//	PlaneMapToDisparityMap(coeffsL, dispL);
//	PlaneMapToDisparityMap(coeffsR, dispR);


	// Weighted median filtering 
	int maxround = 1;
	bool useInvalidPixels = true;
	for (int round = 0; round < maxround; round++) {

		PlaneMapToDisparityMap(coeffsL, dispL);
		PlaneMapToDisparityMap(coeffsR, dispR);
		CrossCheck(dispL, dispR, validL, validR);

		//if (round + 1 == maxround) {
		//	useInvalidPixels = false;
		//}
#pragma omp parallel for
		for (int y = 0; y < nrows; y++) {
			//if (y % 10 == 0) { printf("median filtering row %d\n", y); }
			for (int x = 0; x < ncols; x++) {
				if (!validL[y][x]){
					MCImg<float> wL(patch_w, patch_w, 1, weightsL.line(y * ncols + x));
					WeightedMedianFilter(y, x, dispL, wL, validL, useInvalidPixels);
				}
				if (!validR[y][x]){
					MCImg<float> wR(patch_w, patch_w, 1, weightsR.line(y * ncols + x));
					WeightedMedianFilter(y, x, dispR, wR, validR, useInvalidPixels);
				}
			}
		}
	}
}