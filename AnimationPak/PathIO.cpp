
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

void PathIO::SaveText(std::string content_string, std::string filename)
{
	std::ofstream* f = new std::ofstream();
	f->open(filename);

	*f << content_string;

	f->close();
	delete f;

}

/*
file format:
	[ 1 num_all_points ]
	[ 2 num_layer ]
	[ 3 num_art ]
	[ 4 num_points_per_layer ]
	[ 5 num_boundary_points_per_layer ]
	[ 6 num_triangles_per_layer ]
	7  x0 y0 z0 x1 y1 z1 z0 x2 y2 z2 ...  % all points (all layers)
	8  idx_0_0 idx_0_1 idx_0_2 ...        % layer triangles (all layers)
	9  idx_0_0 idx_0_1 ...                % neg space edges  (all layers)
	10 x0 y0 z0 x1 y1 z1 x2 y2 z2 ...	   % art 1 (first layer only)
	   x0 y0 z0 x1 y1 z1 x2 y2 z2 ...	   % art 2 (first layer only)
	   ...
	11 idx0 idx1 idx2 ... % art to tri
	   idx0 idx1 idx2 ... % art to tri
	   ...
	12 u0 v0 w0...        % bary of art 1
	   u0 v0 w0...        % bary of art 2
	   ...
	13 r,g,b r,g,b r,g,b ...  % foreground colors for arts
	13 r,g,b r,g,b r,g,b ...  % background colors for arts
*/
void PathIO::SaveAnimatedElement(AnElement elem, std::string filename)
{
	std::ofstream* f = new std::ofstream();
	f->open(filename);

	// ----- 1 num_all_points ----- 
	*f << elem._massList.size() << "\n";

	// ----- 2 num_layer ----- 
	*f << SystemParams::_num_layer << "\n";

	// ----- 3 num arts -----
	*f << elem._arts.size() << "\n";

	// ----- 4 num_points_per_layer----- 
	*f << elem._numPointPerLayer << "\n";

	// ----- 5 num_boundary_points_per_layer----- 
	*f << elem._numBoundaryPointPerLayer << "\n";

	// ----- 6 num_triangles_per_layer----- 
	*f << elem._numTrianglePerLayer << "\n";

	// ----- mass pos ----- 
	for (int a = 0; a < elem._massList.size(); a++)
	{
		*f << std::setprecision(20) << elem._massList[a]._pos._x << " " << std::setprecision(20) << elem._massList[a]._pos._y << " " << std::setprecision(20) << elem._massList[a]._pos._z;
		if (a < elem._massList.size() - 1) { *f << " "; }
	}
	*f << "\n";

	// ----- layer triangles ----- 
	for (int a = 0; a < elem._triangles.size(); a++)
	{
		*f << elem._triangles[a].idx0 << " "
			<< elem._triangles[a].idx1 << " "
			<< elem._triangles[a].idx2;
		if (a < elem._triangles.size() - 1) { *f << " "; }
	}
	*f << "\n";

	// _neg_space_springs
	for (int a = 0; a < elem._neg_space_springs.size(); a++)
	{
		*f << elem._neg_space_springs[a]._index0 << " " << elem._neg_space_springs[a]._index1;
		if (a < elem._neg_space_springs.size() - 1) { *f << " "; }
	}
	*f << "\n";

	// arts
	for (int a = 0; a < elem._arts.size(); a++)
	{
		for (int b = 0; b < elem._arts[a].size(); b++)
		{
			*f << std::setprecision(20)
				<< elem._arts[a][b].x << " "
				<< std::setprecision(20)
				<< elem._arts[a][b].y;
			if (b < elem._arts[a].size() - 1) { *f << " "; }
		}
		*f << "\n";
	}

	// art to triangle indices
	for (int a = 0; a < elem._arts2Triangles.size(); a++)
	{
		for (int b = 0; b < elem._arts2Triangles[a].size(); b++)
		{
			*f << elem._arts2Triangles[a][b];
			if (b < elem._arts2Triangles[a].size() - 1) { *f << " "; }
		}
		*f << "\n";
	}

	// barycentric coordinates
	for (int a = 0; a < elem._baryCoords.size(); a++)
	{
		for (int b = 0; b < elem._baryCoords[a].size(); b++)
		{
			*f << std::setprecision(20)
				<< elem._baryCoords[a][b]._u << " "
				<< std::setprecision(20)
				<< elem._baryCoords[a][b]._v << " "
				<< std::setprecision(20)
				<< elem._baryCoords[a][b]._w;
			if (b < elem._baryCoords[a].size() - 1) { *f << " "; }
		}
		if (a < elem._baryCoords.size() - 1) { *f << "\n"; }
	}


	*f << "\n";

	// foreground colors for arts
	for (int a = 0; a < elem._art_f_colors.size(); a++)
	{
		MyColor col = elem._art_f_colors[a];
		*f << col._r << "," << col._g << "," << col._b;
		if (a < elem._art_f_colors.size() - 1) { *f << " "; }
	}
	*f << "\n";

	// background colors for arts
	for (int a = 0; a < elem._art_b_colors.size(); a++)
	{
		MyColor col = elem._art_b_colors[a];
		*f << col._r << "," << col._g << "," << col._b;
		if (a < elem._art_b_colors.size() - 1) { *f << " "; }
	}

	f->close();

	delete f;
}

/*
file format:
	[ 1 num_all_points ]
	[ 2 num_layer ]
	[ 3 num_art ]
	[ 4 num_points_per_layer ]
	[ 5 num_boundary_points_per_layer ]
	[ 6 num_triangles_per_layer ]
	7  x0 y0 z0 x1 y1 z1 z0 x2 y2 z2 ...  % all points (all layers)
	8  idx_0_0 idx_0_1 idx_0_2 ...        % layer triangles (all layers)
	9  idx_0_0 idx_0_1 ...                % neg space edges  (all layers)
	10 x0 y0 z0 x1 y1 z1 x2 y2 z2 ...	   % art 1 (first layer only)
		x0 y0 z0 x1 y1 z1 x2 y2 z2 ...	   % art 2 (first layer only)
		...
	11 idx0 idx1 idx2 ... % art to tri
		idx0 idx1 idx2 ... % art to tri
		...
	12 u0 v0 w0...        % bary of art 1
		u0 v0 w0...        % bary of art 2
		...
	13 r,g,b r,g,b r,g,b ...  % foreground colors for arts
	13 r,g,b r,g,b r,g,b ...  % background colors for arts
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

	// 13 foreground colors
	std::string line13;
	std::getline(myfile, line13);
	std::vector<std::string> arrayTemp13 = UtilityFunctions::Split(line13, ' ');
	for (int a = 0; a < arrayTemp13.size(); a++)
	{
		std::vector<std::string> col_array = UtilityFunctions::Split(arrayTemp13[a], ',');
		elem._art_f_colors.push_back(MyColor(stoi(col_array[0]), stoi(col_array[1]), stoi(col_array[2])));
	}


	// 14 background colors
	std::string line14;
	std::getline(myfile, line14);
	std::vector<std::string> arrayTemp14 = UtilityFunctions::Split(line14, ' ');
	for (int a = 0; a < arrayTemp14.size(); a++)
	{
		std::vector<std::string> col_array = UtilityFunctions::Split(arrayTemp14[a], ',');
		elem._art_b_colors.push_back(MyColor(stoi(col_array[0]), stoi(col_array[1]), stoi(col_array[2])));
	}

	// name
	std::vector<std::string> string_elems = UtilityFunctions::Split(filename, '\\');
	std::vector<std::string> name_only_array = UtilityFunctions::Split(string_elems[string_elems.size() - 1], '.');
	std::string name_only_no_ext = name_only_array[0];
	elem._name = name_only_no_ext;

	// stuff
	elem._numPointPerLayer = num_points_per_layer;
	elem._numBoundaryPointPerLayer = num_boundary_points_per_layer;
	elem._numTrianglePerLayer = num_tri_per_layer;

	return elem;
}

void PathIO::LoadContainerWithHole(std::string filename,
								   std::vector<std::vector<A2DVector>>& boundaries,
								   std::vector<std::vector<A2DVector>>& holes)
{
	std::ifstream myfile(filename);

	// first line only
	std::string firstLine;
	std::getline(myfile, firstLine);
	int numRegion = (int)std::stoi(firstLine);
	boundaries.reserve(numRegion);

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
		int pathType = std::stoi(metadataArray[2]); // path type

		// read path
		if (pathArray.size() < 2) { continue; }
		size_t halfLength = pathArray.size() / 2;
		for (size_t a = 0; a < halfLength; a++)
		{
			int idx = a * 2;
			double x = std::stof(pathArray[idx]);
			double y = std::stof(pathArray[idx + 1]);
			aPath.push_back(A2DVector(x, y));
		}

		if (pathType == 7)
		{
			holes.push_back(aPath);
		}
		else
		{
			boundaries.push_back(aPath);
		}
	}

	myfile.close();

	//return boundaries;
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

/*
file format
	1 num_layer
	2 num_path
	3 num_position
	x0 y0 z0 l0.... % path 1
	...
	x0 y0 x1 y1... % all positions
*/
/*
std::vector<A2DVector> _positions;
std::vector < std::vector<A3DVector>> _paths; //[a path][layer]
*/
void PathIO::LoadScenes(std::vector <std::vector<A3DVector>>& paths,
					   std::vector<std::vector<int>>& layer_indices,
	                   std::vector<A2DVector>& positions, 
					   std::string filename)
{
	std::ifstream myfile(filename);

	// 1 num_layer
	std::string line1;
	std::getline(myfile, line1);
	int num_layer = std::stoi(line1);

	// 2 num_path
	std::string line2;
	std::getline(myfile, line2);
	int num_paths = std::stoi(line2);

	// 3 num_position
	std::string line3;
	std::getline(myfile, line3);
	int num_positions = std::stoi(line3);

	/*
	x0 y0 z0 l0.... % path 1
	...
	*/
	for (int a = 0; a < num_paths; a++)
	{
		std::vector<A3DVector> aPath;
		std::vector<int> indices;

		std::string line4;
		std::getline(myfile, line4);
		std::vector<std::string> arrayTemp4 = UtilityFunctions::Split(line4, ' ');
		for (int a = 0; a < arrayTemp4.size(); a += 4)
		{
			float x = std::stof(arrayTemp4[a]);
			float y = std::stof(arrayTemp4[a + 1]);
			float z = std::stof(arrayTemp4[a + 2]);
			int l = std::stoi(arrayTemp4[a + 3]);

			aPath.push_back(A3DVector(x, y, z));
			indices.push_back(l);
		}

		paths.push_back(aPath);
		layer_indices.push_back(indices);
	}

	// x0 y0 x1 y1... % all positions
	std::string line5;
	std::getline(myfile, line5);
	std::vector<std::string> arrayTemp5 = UtilityFunctions::Split(line5, ' ');
	size_t halfLength = arrayTemp5.size() / 2;
	for (int a = 0; a < halfLength; a++)
	{
		int idx = a * 2;
		float xpos = std::stof(arrayTemp5[idx]);
		float ypos = std::stof(arrayTemp5[idx + 1]);

		positions.push_back(A2DVector(xpos, ypos));
	}

	myfile.close();
}
