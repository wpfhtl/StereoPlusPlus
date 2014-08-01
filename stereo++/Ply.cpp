#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


static cv::Vec3f CrossProduct(cv::Vec3f& u, cv::Vec3f& v)
{
	cv::Vec3f w;
	// u /cross v = (u2v3i +u3v1j + u1v2k) - (u3v2i + u1v3j + u2v1k)
	w[0] = u[1] * v[2] - u[2] * v[1];
	w[1] = u[2] * v[0] - u[0] * v[2];
	w[2] = u[0] * v[1] - u[1] * v[0];
	float norm = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
	norm = std::max(0.01f, norm);
	w[0] /= norm;
	w[1] /= norm;
	w[2] /= norm;

	return w;
}

void SaveDisparityToPly(cv::Mat &disp, cv::Mat& img, float maxDisp, 
	std::string workingDir, std::string plyFilePath, cv::Mat &validPixelMap = cv::Mat())
{
	int numRows = img.rows, numCols = img.cols;
	if (validPixelMap.empty()) {
		validPixelMap = cv::Mat(disp.size(), CV_8UC1, cv::Scalar(255));
	}
	int numValidPixels = cv::countNonZero(validPixelMap);

	const float f = 3740;
	const float B = 160;
	const float cutoff = 0;
	float dmin = 270;

	FILE *fid = fopen((workingDir + "/dmin.txt").c_str(), "r");
	if (fid != NULL) {
		fscanf(fid, "%f", &dmin);
		fclose(fid);
	}
	printf("dmin = %f\n", dmin);

	cv::Mat X(disp.size(), CV_32FC1);
	cv::Mat Y(disp.size(), CV_32FC1);
	cv::Mat Z(disp.size(), CV_32FC1);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			float d = disp.at<float>(y, x);
			d = std::max(d, cutoff);
			d = std::min(d, (float)maxDisp);
			d += dmin;
			Z.at<float>(y, x) = f * B / d;
			X.at<float>(y, x) = x * Z.at<float>(y, x) / f;
			Y.at<float>(y, x) = y * Z.at<float>(y, x) / f;
		}
	}


	fid = fopen(plyFilePath.c_str(), "w");
	fprintf(fid, "ply\nformat ascii 1.0 \nelement vertex %d \n", numValidPixels);
	fprintf(fid, "property float x \nproperty float y \nproperty float z \nproperty float nx \nproperty float ny \nproperty float nz \n");
	//fprintf(fid, "property uchar diffuse_red \nproperty uchar diffuse_green \nproperty uchar diffuse_blue \n");
	fprintf(fid, "property uchar red \nproperty uchar green \nproperty uchar blue \n");
	/*if (facets.size() > 0) {
		fprintf(fid, "element face %d\nproperty list uchar int vertex_indices\n", facets.size());
	}*/
	fprintf(fid, "end_header \n");

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {

			if (!validPixelMap.at<unsigned char>(y, x)) {
				continue;
			}

			cv::Vec3b c = img.at<cv::Vec3b>(y, x);
			cv::Vec3f A, B, C;
			A[0] = X.at<float>(y, x);
			A[1] = Y.at<float>(y, x);
			A[2] = Z.at<float>(y, x);
			B = A;
			C = A;
			if (x + 1 < numCols) {
				B[0] = X.at<float>(y, x + 1);
				B[1] = Y.at<float>(y, x + 1);
				B[2] = Z.at<float>(y, x + 1);
			}
			if (y + 1 < numRows) {
				C[0] = X.at<float>(y + 1, x);
				C[1] = Y.at<float>(y + 1, x);
				C[2] = Z.at<float>(y + 1, x);
			}
			cv::Vec3f N = CrossProduct(C - A, B - A);
			fprintf(fid, "%f %f %f %f %f %f %d %d %d\n", A[0], A[1], A[2], N[0], N[1], N[2], c[2], c[1], c[0]);
		}
	}

	//int ind[3];
	//for (int i = 0; i < facets.size(); i++) {
	//	Facet triangle = facets[i];
	//	for (int j = 0; j < 3; j++) {
	//		int y = triangle.vertex[j].y;
	//		int x = triangle.vertex[j].x;
	//		inline float CLAMP(float x, float a, float b);
	//		x = CLAMP(x, 0, numCols - 1);
	//		y = CLAMP(y, 0, numRows - 1);
	//		ind[j] = y * numCols + x;
	//	}
	//	fprintf(fid, "%d %d %d %d\n", 3, ind[2], ind[1], ind[0]);
	//}

	fclose(fid);
}