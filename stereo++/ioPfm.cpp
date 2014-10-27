#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



#include <math.h>
#include <string.h>

typedef  unsigned char uchar;


bool FileExist(std::string filePath)
{
	if (FILE *file = fopen(filePath.c_str(), "r")) {
		fclose(file);
		return true;
	}
	else {
		return false;
	}
}

// check whether machine is little endian
int littleendian()
{
	int intval = 1;
	uchar *uval = (uchar *)&intval;
	return uval[0] == 1;
}


void skip_comment(FILE *fp)
{
	// skip comment lines in the headers of pnm files

	char c;
	while ((c = getc(fp)) == '#')
	while (getc(fp) != '\n');
	ungetc(c, fp);
}

void skip_space(FILE *fp)
{
	// skip white space in the headers or pnm files

	char c;
	do {
		c = getc(fp);
	} while (c == '\n' || c == ' ' || c == '\t' || c == '\r');
	ungetc(c, fp);
}


void read_header(FILE *fp, const char *imtype, char c1, char c2,
	int *width, int *height, int *nbands, int thirdArg)
{
	// read the header of a pnmfile and initialize width and height

	char c;

	if (getc(fp) != c1 || getc(fp) != c2)
		printf("ReadFilePGM: wrong magic code for %s file", imtype);
	skip_space(fp);
	skip_comment(fp);
	skip_space(fp);
	fscanf(fp, "%d", width);
	skip_space(fp);
	fscanf(fp, "%d", height);
	if (thirdArg) {
		skip_space(fp);
		fscanf(fp, "%d", nbands);
	}
	// skip SINGLE newline character after reading image height (or third arg)
	c = getc(fp);
	if (c == '\r')      // <cr> in some files before newline
		c = getc(fp);
	if (c != '\n') {
		if (c == ' ' || c == '\t' || c == '\r')
			printf("newline expected in file after image height");
		else
			printf("whitespace expected in file after image height");
	}
}



// 1-band PFM image, see http://netpbm.sourceforge.net/doc/pfm.html
// 3-band not yet supported
cv::Mat ReadFilePFM(std::string filePath)
{
	// Open the file and read the header
	FILE *fp = fopen(filePath.c_str(), "rb");
	if (fp == 0)
		printf("ReadFilePFM: could not open %s", filePath.c_str());

	int width, height, nBands;
	read_header(fp, "PFM", 'P', 'f', &width, &height, &nBands, 0);

	skip_space(fp);

	float scalef;
	fscanf(fp, "%f", &scalef);  // scale factor (if negative, little endian)

	// skip SINGLE newline character after reading third arg
	char c = getc(fp);
	if (c == '\r')      // <cr> in some files before newline
		c = getc(fp);
	if (c != '\n') {
		if (c == ' ' || c == '\t' || c == '\r')
			printf("newline expected in file after scale factor");
		else
			printf("whitespace expected in file after scale factor");
	}

	// 	// Set the image shape
	// 	CShape sh(width, height, 1);
	// 
	// 	// Allocate the image if necessary
	// 	img.ReAllocate(sh);

	cv::Mat output(height, width, CV_32FC1);
	float *imDataOut = (float*)output.data;

	int littleEndianFile = (scalef < 0);
	int littleEndianMachine = littleendian();
	int needSwap = (littleEndianFile != littleEndianMachine);
	//printf("endian file = %d, endian machine = %d, need swap = %d\n", 
	//       littleEndianFile, littleEndianMachine, needSwap);

	for (int y = height - 1; y >= 0; y--) { // PFM stores rows top-to-bottom!!!!
		int n = width;
		// 		float* ptr = (float *)img.PixelAddress(0, y, 0);
		float *ptr = imDataOut + y * width + 0;
		if ((int)fread(ptr, sizeof(float), n, fp) != n)
			printf("ReadFilePFM(%s): file is too short", filePath.c_str());

		if (needSwap) { // if endianness doesn't agree, swap bytes
			// 			uchar* ptr = (uchar *)img.PixelAddress(0, y, 0);
			unsigned char *ptr = (unsigned char*)(imDataOut + y * width + 0);
			int x = 0;
			uchar tmp = 0;
			while (x < n) {
				tmp = ptr[0]; ptr[0] = ptr[3]; ptr[3] = tmp;
				tmp = ptr[1]; ptr[1] = ptr[2]; ptr[2] = tmp;
				ptr += 4;
				x++;
			}
		}
	}
	if (fclose(fp))
		printf("ReadFilePGM(%s): error closing file", filePath.c_str());

	return output;
}





//void WriteFilePFM(float *img, int width, int height, const char* filename, float scalefactor = 1 / 255.0)
void WriteFilePFM(std::string filepath, cv::Mat &img)
{
	float scalefactor = 1;
	// Write a PFM file

	// Open the file
	FILE *stream = fopen(filepath.c_str(), "wb");
	if (stream == 0)
		printf("WriteFilePFM: could not open %s", filepath.c_str());

	// sign of scalefact indicates endianness, see pfms specs
	if (littleendian())
		scalefactor = -scalefactor;

	// write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
	int width = img.cols;
	int height = img.rows;
	fprintf(stream, "Pf\n%d %d\n%f\n", width, height, scalefactor);

	int n = width;
	// write rows -- pfm stores rows in inverse order!
	for (int y = height - 1; y >= 0; y--) {
		// 	float* ptr = (float *)img.PixelAddress(0, y, 0);
		float* ptr = (float *)img.data + y * width;
		if ((int)fwrite(ptr, sizeof(float), n, stream) != n)
			printf("WriteFilePFM(%s): file is too short", filepath.c_str());
	}

	// close file
	if (fclose(stream))
		printf("WriteFilePFM(%s): error closing file", filepath.c_str());
}


