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

MCImg<float>				gDsiL;
MCImg<float>				gDsiR;
MCImg<float>				gSimWeightsL;
MCImg<float>				gSimWeightsR;
MCImg<SimVector>			gSimVecsL;
MCImg<SimVector>			gSimVecsR;

enum CostAggregationType	{ GRID, TOP50 };
enum MatchingCostType		{ ADGRADIENT, ADCENSUS };
CostAggregationType			gCostAggregationType;
MatchingCostType			gMatchingCostType;

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
int							SEGMENT_LEN;





struct GlobalParamsInitializer
{
	GlobalParamsInitializer();
}
autoInitObj;



GlobalParamsInitializer::GlobalParamsInitializer()
{
	std::string paramFilePath = "d:/data/stereo_params.txt";
	FILE *fid = fopen(paramFilePath.c_str(), "r");
	ASSERT(fid != NULL);

	char keyStr[1024], valStr[1024], lineBuf[1024];
	while (fgets(lineBuf, 1023, fid) != NULL) {
		if (std::string(lineBuf) == "\n") {
			continue;
		}
		sscanf(lineBuf, "%s%s", keyStr, valStr);

		if (std::string(keyStr) == "TAU1") {
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
		else if (std::string(keyStr) == "SEGMENT_LEN") {
			sscanf(valStr, "%d", &SEGMENT_LEN);
			printf("%20s = %d\n", "SEGMENT_LEN", SEGMENT_LEN);
		}
	}
	
	fclose(fid);
}
