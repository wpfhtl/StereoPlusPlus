#include <string>
#include "MCImg.h"
#include "StereoAPI.h"
#include "ReleaseAssert.h"



class GlobalParams
{
	// A singleton class to store global parameters.
	// We should avoid to use any global variables.
	// Put all the global variables in this class instead.
public:
	static GlobalParams& GetInstance()
	{
		static GlobalParams    instance;	// Guaranteed to be destroyed.
		// Instantiated on first use.
		return instance;
	}
private:
	GlobalParams() {};						// Constructor? (the {} brackets) are needed here.
	// Dont forget to declare these two. You want to make sure they
	// are unaccessable otherwise you may accidently get copies of
	// your singleton appearing.
	GlobalParams(GlobalParams const&);		// Don't Implement
	void operator=(GlobalParams const&);	// Don't implement

public:
};



//MCImg<float>				gDsiL;
//MCImg<float>				gDsiR;
MCImg<unsigned short>		gDsiL;
MCImg<unsigned short>		gDsiR;
MCImg<float>				gSimWeightsL;
MCImg<float>				gSimWeightsR;
MCImg<SimVector>			gSimVecsL;
MCImg<SimVector>			gSimVecsR;

cv::Mat gImLabL, gImLabR, gImGradL, gImGradR;
cv::Mat gLabelMapL, gLabelMapR;
cv::Mat gSobelImgL, gSobelImgR, gCensusImgL, gCensusImgR;
int gNumDisps = 0;
int NUM_PREFERED_REGIONS = 4000;
std::string gFilePathOracleL, gFilePathOracleR;
cv::Mat gDispSGML, gDispSGMR;
std::string midd3Resolution, midd3TestCaseId;
std::string midd3BasePath = "D:\\data\\MiddEval3";
std::string kittiTestCaseId;
std::vector<std::vector<cv::Point2i>> gSegPixelListsL, gSegPixelListsR;
std::vector<std::vector<int>> segAnchorIndsL, segAnchorIndsR;
int VISUALIZE_EVAL, DO_EVAL;


enum CostAggregationType	{ GRID, TOP50 };
enum MatchingCostType		{ ADGRADIENT, ADCENSUS };
CostAggregationType			gCostAggregationType;
MatchingCostType			gMatchingCostType;
int							INTERP_ONLINE;


float						TAU1;
float						TAU2;
float						TAU3;
float						TANGENT_CUTOFF;
float						TANGENT_LAMBDA;

int							PATCHRADIUS;
int							PATCHWIDTH;
float						GRANULARITY;

float						SIMILARITY_GAMMA;
float						ISING_CUTOFF;
float						ISING_LAMBDA;
float						ISING_GAMMA;

std::string					ROOTFOLDER;

bool						USE_CONVEX_BP;
int							MAX_PATCHMATCH_ITERS;

float						COLORGRADALPHA;
float						COLORMAXDIFF;
float						GRADMAXDIFF;

int							PROGRAM_ENTRY;

float						POSTALIGN_TAU1;
float						POSTALIGN_TAU2;

float						ARAP_LAMBDA;
float						ARAP_SIGMA;
float						ARAP_THETASCALE;
float						ARAP_THETAOFFSET;
int							ARAP_MAX_ITERS;
int							ARAP_NUM_ANCHORS;
int							ARAP_INIT_PATCHMATCH_ITERS = 8;
int							SEGMENT_LEN;
float						ARAP_CENSUS_WEIGHT;
float						SLIC_COMPACTNESS = 20.f;
int							MATCHINGCOST_STRIDE = 1;
float						ARAP_NORMALDIFF_TRUNCATION = 4;
float						ARAP_DISPDIFF_TRUNCATION = 500;


//struct GlobalParamsInitializer
//{
//	GlobalParamsInitializer();
//}
//autoInitObj;

  

//GlobalParamsInitializer::GlobalParamsInitializer()
void ReadStereoParameters(std::string filePathStereoParams = "")
{
	if (filePathStereoParams == "") {
		filePathStereoParams = "d:/data/stereo_params.txt";
	}
	printf("filePathStereoParams: %s\n", filePathStereoParams.c_str());
	FILE *fid = fopen(filePathStereoParams.c_str(), "r");
	ASSERT(fid != NULL);

	char keyStr[1024], valStr[1024], lineBuf[1024];
	while (fgets(lineBuf, 1023, fid) != NULL) {
		if (std::string(lineBuf) == "\n") {
			continue;
		}
		sscanf(lineBuf, "%s%s", keyStr, valStr);


		if (std::string(keyStr) == "INTERP_ONLINE") {
			sscanf(valStr, "%d", &INTERP_ONLINE);
			printf("%20s = %d\n", "INTERP_ONLINE", INTERP_ONLINE);
		}
		else if (std::string(keyStr) == "TAU1") {
			sscanf(valStr, "%f", &TAU1);
			printf("%20s = %f\n", "TAU1", TAU1);
		}
		else if (std::string(keyStr) == "TAU2") {
			sscanf(valStr, "%f", &TAU2);
			printf("%20s = %f\n", "TAU2", TAU2);
		}
		else if (std::string(keyStr) == "TAU3") {
			sscanf(valStr, "%f", &TAU3);
			printf("%20s = %f\n", "TAU3", TAU3);
		}
		else if (std::string(keyStr) == "TANGENT_CUTOFF") {
			sscanf(valStr, "%f", &TANGENT_CUTOFF);
			printf("%20s = %f\n", "TANGENT_CUTOFF", TANGENT_CUTOFF);
		}
		else if (std::string(keyStr) == "TANGENT_LAMBDA") {
			sscanf(valStr, "%f", &TANGENT_LAMBDA);
			printf("%20s = %f\n", "TANGENT_LAMBDA", TANGENT_LAMBDA);
		}
		else if (std::string(keyStr) == "PATCHRADIUS") {
			sscanf(valStr, "%d", &PATCHRADIUS);
			printf("%20s = %d\n", "PATCHRADIUS", PATCHRADIUS);
		}
		else if (std::string(keyStr) == "PATCHWIDTH") {
			sscanf(valStr, "%d", &PATCHWIDTH);
			printf("%20s = %d\n", "PATCHWIDTH", PATCHWIDTH);
		}
		else if (std::string(keyStr) == "GRANULARITY") {
			sscanf(valStr, "%f", &GRANULARITY);
			printf("%20s = %f\n", "GRANULARITY", GRANULARITY);
		}
		else if (std::string(keyStr) == "SIMILARITY_GAMMA") {
			sscanf(valStr, "%f", &SIMILARITY_GAMMA);
			printf("%20s = %f\n", "SIMILARITY_GAMMA", SIMILARITY_GAMMA);
		}
		else if (std::string(keyStr) == "ISING_CUTOFF") {
			sscanf(valStr, "%f", &ISING_CUTOFF);
			printf("%20s = %f\n", "ISING_CUTOFF", ISING_CUTOFF);
		}
		else if (std::string(keyStr) == "ISING_LAMBDA") {
			sscanf(valStr, "%f", &ISING_LAMBDA);
			printf("%20s = %f\n", "ISING_LAMBDA", ISING_LAMBDA);
		}
		else if (std::string(keyStr) == "ISING_GAMMA") {
			sscanf(valStr, "%f", &ISING_GAMMA);
			printf("%20s = %f\n", "ISING_GAMMA", ISING_GAMMA);
		}
		else if (std::string(keyStr) == "ROOTFOLDER") {
			ROOTFOLDER = std::string(valStr);
			printf("%20s = %s\n", "ROOTFOLDER", ROOTFOLDER.c_str());
		}
		else if (std::string(keyStr) == "COSTAGGREGATION_TYPE") {
			gCostAggregationType 
				= (std::string(valStr) == "GRID" ? GRID : TOP50);
			printf("%20s = %d\n", "COSTAGGREGATION_TYPE", gCostAggregationType);
		}
		else if (std::string(keyStr) == "MATCHINGCOST_TYPE") {
			gMatchingCostType 
				= (std::string(valStr) == "ADCENSUS" ? ADCENSUS : ADGRADIENT);
			printf("%20s = %d\n", "MATCHINGCOST_TYPE", gMatchingCostType);
		}
		else if (std::string(keyStr) == "MAX_PATCHMATCH_ITERS") {
			sscanf(valStr, "%d", &MAX_PATCHMATCH_ITERS);
			printf("%20s = %d\n", "MAX_PATCHMATCH_ITERS", MAX_PATCHMATCH_ITERS);
		}
		else if (std::string(keyStr) == "USE_CONVEX_BP") {
			USE_CONVEX_BP = (std::string(valStr) == "true" || std::string(valStr) == "1");
			printf("%20s = %d\n", "USE_CONVEX_BP", USE_CONVEX_BP);
		}
		else if (std::string(keyStr) == "COLORGRADALPHA") {
			sscanf(valStr, "%f", &COLORGRADALPHA);
			printf("%20s = %f\n", "COLORGRADALPHA", COLORGRADALPHA);
		}
		else if (std::string(keyStr) == "COLORMAXDIFF") {
			sscanf(valStr, "%f", &COLORMAXDIFF);
			printf("%20s = %f\n", "COLORMAXDIFF", COLORMAXDIFF);
		}
		else if (std::string(keyStr) == "GRADMAXDIFF") {
			sscanf(valStr, "%f", &GRADMAXDIFF);
			printf("%20s = %f\n", "GRADMAXDIFF", GRADMAXDIFF);
		}
		else if (std::string(keyStr) == "PROGRAM_ENTRY") {
			sscanf(valStr, "%d", &PROGRAM_ENTRY);
			printf("%20s = %d\n", "PROGRAM_ENTRY", PROGRAM_ENTRY);
		}
		else if (std::string(keyStr) == "POSTALIGN_TAU1") {
			sscanf(valStr, "%f", &POSTALIGN_TAU1);
			printf("%20s = %f\n", "POSTALIGN_TAU1", POSTALIGN_TAU1);
		}
		else if (std::string(keyStr) == "POSTALIGN_TAU2") {
			sscanf(valStr, "%f", &POSTALIGN_TAU2);
			printf("%20s = %f\n", "POSTALIGN_TAU2", POSTALIGN_TAU2);
		}
		else if (std::string(keyStr) == "ARAP_LAMBDA") {
			sscanf(valStr, "%f", &ARAP_LAMBDA);
			printf("%20s = %f\n", "ARAP_LAMBDA", ARAP_LAMBDA);
		}
		else if (std::string(keyStr) == "ARAP_SIGMA") {
			sscanf(valStr, "%f", &ARAP_SIGMA);
			printf("%20s = %f\n", "ARAP_SIGMA", ARAP_SIGMA);
		}		
		else if (std::string(keyStr) == "ARAP_THETASCALE") {
			sscanf(valStr, "%f", &ARAP_THETASCALE);
			printf("%20s = %f\n", "ARAP_THETASCALE", ARAP_THETASCALE);
		}
		else if (std::string(keyStr) == "ARAP_THETAOFFSET") {
			sscanf(valStr, "%f", &ARAP_THETAOFFSET);
			printf("%20s = %f\n", "ARAP_THETAOFFSET", ARAP_THETAOFFSET);
		}
		else if (std::string(keyStr) == "ARAP_MAX_ITERS") {
			sscanf(valStr, "%d", &ARAP_MAX_ITERS);
			printf("%20s = %d\n", "ARAP_MAX_ITERS", ARAP_MAX_ITERS);
		}
		else if (std::string(keyStr) == "ARAP_NUM_ANCHORS") {
			sscanf(valStr, "%d", &ARAP_NUM_ANCHORS);
			printf("%20s = %d\n", "ARAP_NUM_ANCHORS", ARAP_NUM_ANCHORS);
		}
		else if (std::string(keyStr) == "SEGMENT_LEN") {
			sscanf(valStr, "%d", &SEGMENT_LEN);
			printf("%20s = %d\n", "SEGMENT_LEN", SEGMENT_LEN);
		}
		else if (std::string(keyStr) == "ARAP_CENSUS_WEIGHT") {
			sscanf(valStr, "%f", &ARAP_CENSUS_WEIGHT);
			printf("%20s = %f\n", "ARAP_CENSUS_WEIGHT", ARAP_CENSUS_WEIGHT);
		}
		else if (std::string(keyStr) == "ARAP_INIT_PATCHMATCH_ITERS") {
			sscanf(valStr, "%d", &ARAP_INIT_PATCHMATCH_ITERS);
			printf("%20s = %d\n", "ARAP_INIT_PATCHMATCH_ITERS", ARAP_INIT_PATCHMATCH_ITERS);
		}
		else if (std::string(keyStr) == "SLIC_COMPACTNESS") {
			sscanf(valStr, "%f", &SLIC_COMPACTNESS);
			printf("%20s = %f\n", "SLIC_COMPACTNESS", SLIC_COMPACTNESS);
		}
		else if (std::string(keyStr) == "MATCHINGCOST_STRIDE") {
			sscanf(valStr, "%d", &MATCHINGCOST_STRIDE);
			printf("%20s = %d\n", "MATCHINGCOST_STRIDE", MATCHINGCOST_STRIDE);
		}
		else if (std::string(keyStr) == "ARAP_NORMALDIFF_TRUNCATION") {
			sscanf(valStr, "%f", &ARAP_NORMALDIFF_TRUNCATION);
			printf("%20s = %f\n", "ARAP_NORMALDIFF_TRUNCATION", ARAP_NORMALDIFF_TRUNCATION);
		}
		else if (std::string(keyStr) == "ARAP_DISPDIFF_TRUNCATION") {
			sscanf(valStr, "%f", &ARAP_DISPDIFF_TRUNCATION);
			printf("%20s = %f\n", "ARAP_DISPDIFF_TRUNCATION", ARAP_DISPDIFF_TRUNCATION);
		}
		else if (std::string(keyStr) == "midd3BasePath") {
			midd3BasePath = std::string(valStr);
			printf("%20s = %s\n", "midd3BasePath", midd3BasePath.c_str());
		}
	}
	

	fclose(fid);
}
