/*
================================================================================
Reza Adhitya Saputra
radhitya@uwaterloo.ca
================================================================================
*/

#ifndef PATH_IO_H
#define PATH_IO_H

#include <vector>
#include <cstring>

// read directory
#include "dirent.h"

#include "A2DVector.h"
#include "AnElement.h"

class PathIO
{
public:
	PathIO();
	~PathIO();

	std::vector<std::vector<A2DVector>> LoadElement(std::string filename);

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
	void SaveAnimatedElement(AnElement elem, std::string filename);

	void SaveText(std::string content_string, std::string filename);

	AnElement LoadAnimatedElement(std::string filename);

	std::vector<std::string> LoadFiles(std::string directoryPath); // read directory

	// scripted initial placement
	void LoadScenes(std::vector <std::vector<A3DVector>>& paths,
				   std::vector<std::vector<int>>& layer_indices,
		           std::vector<A2DVector>& positions, 
		           std::string filename);

	void SaveContainerToWavefrontOBJ(std::vector<A2DVector>& container_poly, std::string filename);

	void SaveSceneToWavefrontOBJ(std::vector<AnElement>& elems, std::string filename);

	void SaveSceneToWavefrontOBJ(std::vector<AnElement>& elems, int first_idx, int last_idx, std::string filename);

	void SaveFrontBackFacesToWavefrontOBJBasedOnName(std::vector<AnElement>& elems, std::string containStr, std::string filename);

	void SaveSceneToWavefrontOBJBasedOnName(std::vector<AnElement>& elems, std::string containStr, std::string filename);

	void SaveFrontBackFacesToWavefrontOBJ(std::vector<AnElement>& elems, int first_idx, int last_idx, std::string filename);

	void SaveArtsToWavefrontOBJ(std::vector<std::vector<A3DVector>> _arts, std::string filename);
};

#endif
