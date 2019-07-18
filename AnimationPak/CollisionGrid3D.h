
#ifndef __Collision_Grid_3D__
#define __Collision_Grid_3D__

#include <vector>
#include <cmath>

#include "SystemParams.h"
#include "A3DSquare.h"
#include "A3DVector.h"



// Ogre
#include <Ogre.h>
#include <OgreApplicationContext.h>
#include <OgrePrerequisites.h>
#include <OgreMaterial.h>
#include <OgreMesh.h>
#include <OgreSceneManager.h>

#include "DynamicLines.h"

typedef std::vector<int> GraphIndices;
typedef std::vector<std::vector<int>> TriangleIndices;


class CollisionGrid3D
{
private:
	std::vector<A3DObject*>   _objects;

public:
	
	std::vector<A3DSquare*>   _squares;

	std::vector<GraphIndices> _graph_idx_array; // per squares
	std::vector<TriangleIndices> _triangle_idx_array; // per squares

	std::vector<GraphIndices> _approx_graph_idx_array; // per squares
	std::vector<TriangleIndices> _approx_triangle_idx_array; // per squares

	//std::vector<PairData> _pair_data_array; // 2d vector
	//std::vector<PairData> _approx_pair_data_array; // 2d vector
	

	int   _side_num; // side length of the entire grid (unit: cell)
	float _max_cell_length; // side length of a cell

public:
	CollisionGrid3D();

	//CollisionGrid3D(float cellSize);
	void Init();

	void SetPoint(int idx, A3DVector pos);

	~CollisionGrid3D();	

	void InsertAPoint(float x, float y, float z, int info1, int info2); // make z positive

	void GetClosestPoints(float x, float y, float z, std::vector<A3DVector>& closestPoints);

	void GetClosestObjects(float x, float y, float z, std::vector<A3DObject>& closestObjects);

	void GetGraphIndices2B(float x, float y, float z, std::vector<int>& closestGraphIndices);

	void GetTriangleIndices(float x, float y, float z, TriangleIndices& closestTriIndices);

	void GetData(float x, float y, float z, PairData& pair_data_array, PairData& approx_pair_data_array);

	void PrecomputeData();

	void PrecomputeClosestGraphsAndTriangles();

	void MovePoints();

	int GetSquareIndexFromFloat(float x, float y, float z);

private:
	void GetCellPosition(int& xPos, int& yPos, int& zPos, float x, float y, float z);

	int SquareIndex(int xPos, int yPos, int zPos);

	

public:
	// drawing
	// ---------- Ogre 3D ----------
	//DynamicLines*    _empty_lines;
	//Ogre::SceneNode* _empty_node;

	DynamicLines*    _filled_lines;
	Ogre::SceneNode* _filled_node;

	DynamicLines*    _plus_lines;
	Ogre::SceneNode* _plus_node;

	void InitOgre3D(Ogre::SceneManager* sceneMgr);
	void UpdateOgre3D();

};

#endif
