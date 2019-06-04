
#include "PathIO.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include <sstream> // std::stringstream

#include <sys/stat.h>

#include "UtilityFunctions.h"

/*
================================================================================
================================================================================
*/
PathIO::PathIO()
{
}

/*
================================================================================
================================================================================
*/
PathIO::~PathIO()
{
}

/*
================================================================================
================================================================================
*/
std::vector<std::vector<A2DVector>> PathIO::LoadElement(std::string filename)
{
	std::vector<std::vector<A2DVector>> duhRegions;

	std::ifstream myfile(filename);

	// first line only
	std::string firstLine;
	std::getline(myfile, firstLine);
	int numRegion = (int)std::stoi(firstLine);
	duhRegions.reserve(numRegion);

	// iterate every line until end
	while (!myfile.eof())
	{
		std::vector<A2DVector> aPath;

		std::string line;
		std::getline(myfile, line);
		std::vector<std::string> arrayTemp = UtilityFunctions::Split(line, ' ');
		if (arrayTemp.size() == 0) { continue; }
		
		// TO DO: IMPLEMENT THESE
		if (arrayTemp[0] == "foreground_color_rgb") { continue; }
		else if (arrayTemp[0] == "background_color_rgb") { continue; }

		std::vector<std::string> metadataArray(arrayTemp.begin(), arrayTemp.begin() + 3); // this is metadata
		std::vector<std::string> pathArray(arrayTemp.begin() + 3, arrayTemp.end()); // this is path

		// read metadata
		int whatRegionDude = (int)std::stoi(metadataArray[0]);
		//aPath.isClosed = (int)std::stoi(metadataArray[1]);		// is it closed
		//aPath.pathType = (PathType)std::stoi(metadataArray[2]); // path type

		// read path
		if (pathArray.size() < 2) { continue; }
		size_t halfLength = pathArray.size() / 2;
		for (size_t a = 0; a < halfLength; a++)
		{
			int idx = a * 2;
			double x = std::stod(pathArray[idx]);
			double y = std::stod(pathArray[idx + 1]);
			aPath.push_back(A2DVector(x, y));
		}

		duhRegions.push_back(aPath);
	}

	myfile.close();

	return duhRegions;
}