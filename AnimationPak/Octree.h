
#ifndef __Octree_h__
#define __Octree_h__

#include "A3DObject.h"
#include "A3DVector.h"

#include <vector>


class Octree
{
public:
	Octree(float x, float y, float z, float side_width, int level, int max_level, Octree* parent);
	~Octree();

	bool Contains(Octree *child, A3DObject *object);

	void AddTriangle(A3DVector p1, A3DVector p2, A3DVector p3, int info1, int info2);

	void AddObject(A3DObject* object);

	std::vector<A3DObject*> GetInvalidObjectsAndReassign();

private:
	// left hand rule
	// https_//ogrecave.github.io/ogre/api/1.11/tut__first_scene.html
	float _x;
	float _y;
	float _z;

	//float _side_width;
	float _x_max;
	float _y_max;
	float _z_max;


private:
	int _level;
	int _max_level;

	// precomputation
	//int _search_level; 
	//std::vector<A3DObject*> _precomputed_objects; 

	Octree* _parent;

	std::vector<A3DObject*> _objects;
	
	

	/*	
	seen from top

	(-z)
	^
	|
	| 3  4
	| 1  2
	---------> (+x)	

	*/
	
	Octree* _top1;
	Octree* _top2;
	Octree* _top3;
	Octree* _top4;

	/*
	seen from top

	(-z)
	^
	|
	| 3  4
	| 1  2
	---------> (+x)

	*/

	Octree* _bottom1;
	Octree* _bottom2;
	Octree* _bottom3;
	Octree* _bottom4;

	
};

#endif
