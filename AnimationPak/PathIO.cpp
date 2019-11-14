
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
IO
================================================================================
*/
std::vector<std::string> PathIO::LoadFiles(std::string directoryPath)
{
	std::vector<std::string> fileStr;
	/*----------  dirent ----------*/
	DIR *dp;
	struct dirent *ep;
	/*---------- open directory ----------*/
	dp = opendir(directoryPath.c_str());
	if (dp != NULL)
	{
		while (ep = readdir(dp)) { fileStr.push_back(ep->d_name); }
		(void)closedir(dp);
	}
	else { perror("Couldn't open the directory"); }
	return fileStr;
}


/*
file format:
	[ 1 num_all_points ]
	[ 2 num_layer ]
	[ 3 num_art ]
	[ 4 num_points_per_layer ]
	[ 5 num_boundary_points_per_layer ]
	[ 6 num_triangles_per_layer ]
	x0 y0 z0 x1 y1 z1 z0 x2 y2 z2 ...  % all points (all layers)       7
	idx_0_0 idx_0_1 idx_0_2 ...        % layer triangles (all layers)  8
	idx_0_0 idx_0_1 ...                % neg space edges  (all layers) 9
	x0 y0 z0 x1 y1 z1 x2 y2 z2 ...	   % art 1 (first layer only)      10
	x0 y0 z0 x1 y1 z1 x2 y2 z2 ...	   % art 2 (first layer only) 
	...
	idx0 idx1 idx2 ... % art to tri     11
	idx0 idx1 idx2 ... % art to tri
	...
	u0 v0 w0...        % bary of art 1  12
	u0 v0 w0...        % bary of art 2
	...
*/
AnElement PathIO::LoadAnimatedElement(std::string filename)
{
	AnElement elem;
	std::ifstream myfile(filename);

	//  1 num_all_points
	std::string line1;
	std::getline(myfile, line1);
	int num_points = std::stoi(line1);

	// 2 num_layer
	std::string line2;
	std::getline(myfile, line2);
	int num_layer = std::stoi(line2);

	// 3 num_art
	std::string line3;
	std::getline(myfile, line3);
	int num_art = std::stoi(line3);

	// 4 num_points_per_layer
	std::string line4;
	std::getline(myfile, line4);
	int num_points_per_layer = std::stoi(line4);

	// 5 num_boundary_points_per_layer
	std::string line5;
	std::getline(myfile, line5);
	int num_boundary_points_per_layer = std::stoi(line5);

	// 6 num_triangles_per_layer
	std::string line6;
	std::getline(myfile, line6);
	int num_tri_per_layer = std::stoi(line6);

	// 7 x0 y0 z0 x1 y1 z1 z0 x2 y2 z2 ...  % all points (all layers)
	std::string line7;
	std::getline(myfile, line7);
	std::vector<std::string> arrayTemp7 = UtilityFunctions::Split(line7, ' ');
	int massCounter = 0;
	for (int a = 0; a < arrayTemp7.size(); a += 3)
	{		
		float x = std::stof(arrayTemp7[a]);
		float y = std::stof(arrayTemp7[a + 1]);
		float z = std::stof(arrayTemp7[a + 2]);

		int layer_idx = massCounter / num_points_per_layer;
		int per_layer_idx = massCounter % num_points_per_layer;
		bool isBoundary = false;
		if (per_layer_idx < num_boundary_points_per_layer) { isBoundary = true; }

		AMass m(x, 
			    y, 
			    z, 
			    massCounter, // self_idx
			    0,           // parent_idx 
			    layer_idx,   // debug_which_layer
			    isBoundary); // is_boundary

		elem._massList.push_back(m);

		massCounter++;
	}

	// 8 idx_0_0 idx_0_1 idx_0_2 ...        % layer triangles (all layers)
	std::string line8;
	std::getline(myfile, line8);
	std::vector<std::string> arrayTemp8 = UtilityFunctions::Split(line8, ' ');
	for (int a = 0; a < arrayTemp8.size(); a += 3)
	{
		int index0 = std::stoi(arrayTemp8[a]);
		int index1 = std::stoi(arrayTemp8[a + 1]);
		int index2 = std::stoi(arrayTemp8[a + 2]);
		AnIdxTriangle tri(index0, index1, index2);
		elem._triangles.push_back(tri);
	}

	// 9 idx_0_0 idx_0_1 ...                % neg space edges  (all layers)
	std::string line9;
	std::getline(myfile, line9);
	std::vector<std::string> arrayTemp9 = UtilityFunctions::Split(line9, ' ');
	size_t halfLength = arrayTemp9.size() / 2;
	for (int a = 0; a < halfLength; a++)
	{
		int idx = a * 2;
		int index0 = std::stoi(arrayTemp9[idx]);
		int index1 = std::stoi(arrayTemp9[idx + 1]);
		elem._neg_space_springs.push_back(AnIndexedLine(index0, index1));
	}

	/*
	10
	x0 y0 z0 x1 y1 z1 x2 y2 z2 ...	   % art (first layer only)
	...
	*/
	for (int a = 0; a < num_art; a++)
	{
		std::vector<A2DVector> anArt;

		std::string line10;
		std::getline(myfile, line10);
		std::vector<std::string> arrayTemp10 = UtilityFunctions::Split(line10, ' ');
		halfLength = arrayTemp10.size() / 2;
		for (int b = 0; b < halfLength; b++)
		{
			int idx = b * 2;
			float x = std::stof(arrayTemp10[idx]);
			float y = std::stof(arrayTemp10[idx + 1]);
			anArt.push_back(A2DVector(x, y));
		}
		elem._arts.push_back(anArt);
	}

	// 11 art to triangle indices
	for (int a = 0; a < num_art; a++)
	{
		std::vector<int> a2t;

		std::string line11;
		std::getline(myfile, line11);
		std::vector<std::string> arrayTemp11 = UtilityFunctions::Split(line11, ' ');
		for (int b = 0; b < arrayTemp11.size(); b++)
		{
			a2t.push_back(std::stoi(arrayTemp11[b]));
		}
		elem._arts2Triangles.push_back(a2t);
	}

	// 12 barycentric coordinates
	for (int a = 0; a < num_art; a++)
	{
		std::vector<ABary> bCoords;

		std::string line12;
		std::getline(myfile, line12);
		std::vector<std::string> arrayTemp12 = UtilityFunctions::Split(line12, ' ');
		for (int b = 0; b < arrayTemp12.size(); b += 3)
		{
			float u = std::stof(arrayTemp12[b]);
			float v = std::stof(arrayTemp12[b + 1]);
			float w = std::stof(arrayTemp12[b + 2]);
			ABary bary(u, v, w);
			bCoords.push_back(bary);
		}
		elem._baryCoords.push_back(bCoords);
	}

	// stuff
	elem._numPointPerLayer = num_points_per_layer;
	elem._numBoundaryPointPerLayer = num_boundary_points_per_layer;
	elem._numTrianglePerLayer = num_tri_per_layer;

	return elem;
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