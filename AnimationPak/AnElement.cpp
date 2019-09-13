
#include "AnElement.h"
#include "SystemParams.h"
#include "A2DRectangle.h"
#include "UtilityFunctions.h"
#include "PoissonGenerator.h"


#include "OpenCVWrapper.h"
#include "TetWrapper.h"

#include <OgreManualObject.h>
#include <OgreMaterialManager.h>
#include <OgreSceneManager.h>
#include <OgreStringConverter.h>
#include <OgreEntity.h>
#include <OgreMeshManager.h>
#include <OgreHardwareVertexBuffer.h>
#include <OgreHardwareIndexBuffer.h>
#include <OgreSubMesh.h>

#include "NANOFLANNWrapper2D.h"

#include "StuffWorker.h"

template <typename T>
inline const T&
min_of(const T& a, const T& b) {
	return std::min(a, b);
}

template <typename T, typename ...Args>
inline const T&
min_of(const T& a, const T& b, const Args& ...args) 
{
	return min_of(std::min(a, b), args...);
}


AnElement::AnElement()
{
	this->_tubeObject          = 0;
	this->_sceneNode           = 0;
	this->_numPointPerLayer    = 0;
	this->_numBoundaryPointPerLayer = 0;

	this->_layer_center = A2DVector(250, 250);

	this->_scale = 1.0f;
	this->_maxScale = SystemParams::_element_max_scale;
	//this->_uniqueMaterial = false;
	this->_predefined_time_path = false;

	_tempTri3.push_back(A3DVector());
	_tempTri3.push_back(A3DVector());
	_tempTri3.push_back(A3DVector());

	_z_pos_array = std::vector<float>(SystemParams::_num_layer, 0);
}

AnElement::~AnElement()
{
	// still can't create proper destructor ???
	// maybe they're automatically deleted???
	_tubeObject = 0;
	_sceneNode = 0;
	_sceneMgr = 0;
	_material.reset();


	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		_insideFlags.push_back(false);
	}
	/*if (_tubeObject)
	{
	if (_tubeObject->getParentSceneNode())
	_tubeObject->getParentSceneNode()->detachObject(_tubeObject);

	_sceneMgr->destroyManualObject(_tubeObject);
	}

	_material.reset();

	if (_sceneNode)
	{
	_sceneNode->removeAndDestroyAllChildren();
	_sceneNode->getParentSceneNode()->removeAndDestroyChild(_sceneNode->getName());
	_sceneNode = 0;
	}*/

}

void AnElement::UpdatePerLayerInsideFlags()
{

	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		_insideFlags[a] = false;
	}


	for (int a = 0; a < _massList.size(); a++)
	{
		//int layer_idx = _massList[a]._layer_idx;
		//_z_pos_array[layer_idx] += _massList[a]._pos._z;
		if (_massList[a]._is_inside)
		{
			_insideFlags[_massList[a]._layer_idx] = true;
		}
	}
}

void AnElement::UpdateZConstraint()
{
	// reset
	std::fill(_z_pos_array.begin(), _z_pos_array.end(), 0);

	for (int a = 0; a < _massList.size(); a++)
	{
		int layer_idx = _massList[a]._layer_idx;
		_z_pos_array[layer_idx] += _massList[a]._pos._z;
	}

	float numPt = _numPointPerLayer;
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		_z_pos_array[a] /= numPt;
	}

	for (int a = 0; a < _massList.size(); a++)
	{
		int layer_idx = _massList[a]._layer_idx;
		_massList[a]._pos._z = _z_pos_array[layer_idx];
	}
}

void  AnElement::RotateXY(float radAngle)
{
	for (int a = 0; a < _massList.size(); a++)
	{
		A2DVector pos = _massList[a]._pos.GetA2DVector();
		A2DVector rotPos = UtilityFunctions::Rotate(pos, _layer_center, radAngle);
		_massList[a]._pos._x = rotPos.x;
		_massList[a]._pos._y = rotPos.y;
	}

	/*for (int a = 0; a < _interp_massList.size(); a++)
	{
		A2DVector pos = _interp_massList[a]._pos.GetA2DVector();
		A2DVector rotPos = UtilityFunctions::Rotate(pos, _layer_center, radAngle);
		_interp_massList[a]._pos._x = rotPos.x;
		_interp_massList[a]._pos._y = rotPos.y;
	}*/

	ResetSpringRestLengths();
}

void AnElement::ScaleXY(float scVal)
{
	for (int a = 0; a < _massList.size(); a++)
	{
		A3DVector pos = _massList[a]._pos;
		_massList[a]._pos = A3DVector(pos._x * scVal, pos._y * scVal, pos._z);
	}

	/*for (int a = 0; a < _interp_massList.size(); a++)
	{
		A3DVector pos = _interp_massList[a]._pos;
		_interp_massList[a]._pos = A3DVector(pos._x * scVal, pos._y * scVal, pos._z);
	}*/

	ResetSpringRestLengths();
}

void AnElement::TranslateXY(float x, float y)
{

	for (int a = 0; a < _massList.size(); a++)
	{
		A3DVector pos = _massList[a]._pos;
		_massList[a]._pos = A3DVector(pos._x + x, pos._y + y, pos._z);
	}

	/*for (int a = 0; a < _interp_massList.size(); a++)
	{
		A3DVector pos = _interp_massList[a]._pos;
		_interp_massList[a]._pos = A3DVector(pos._x + x, pos._y + y, pos._z);
	}*/

	ResetSpringRestLengths();
}

void AnElement::CreateDockPoint(A2DVector queryPos, A2DVector lockPos, int layer_idx)
{
	int massListIdx = -1;
	float dist = 100000;
	int l1 = layer_idx * _numPointPerLayer;
	int l2 = l1 + _numBoundaryPointPerLayer;
	for (int a = l1; a < l2; a++)
	{
		//if (_massList[a]._layer_idx == layer_idx)
		//{
		float d = _massList[a]._pos.GetA2DVector().Distance(queryPos);
		if (d < dist)
		{
			dist = d;
			massListIdx = a;
		}
		//}
	}

	if (massListIdx == -1)
	{
		std::cout << "error massListIdx == -1\n";
	}

	_massList[massListIdx]._isDocked = true;
	// you probably want the dockpoint be 2D?
	_massList[massListIdx]._dockPoint = A3DVector(lockPos.x, lockPos.y, _massList[massListIdx]._pos._z);

	_dock_mass_idx.push_back(massListIdx);
	//_debug_lines_2->addPoint(Ogre::Vector3(_massList[massListIdx]._pos._x, _massList[massListIdx]._pos._y, _massList[massListIdx]._pos._z));
	//_debug_lines_2->addPoint(Ogre::Vector3(_massList[massListIdx]._dockPoint._x, _massList[massListIdx]._dockPoint._y, _massList[massListIdx]._dockPoint._z));
}

void AnElement::DockEnds(A2DVector startPt2D, A2DVector endPt2D, bool lockEnds)
{
	// ----- stuff -----
	float zGap = SystemParams::_upscaleFactor / (float)(SystemParams::_num_layer - 1);
	A2DVector startPt = _massList[0]._pos.GetA2DVector();
	A2DVector dirVector = startPt.DirectionTo(endPt2D);
	float xyGap = dirVector.Length() / (float)(SystemParams::_num_layer - 1);
	dirVector = dirVector.Norm();

	for (int a = 0; a < _massList.size(); a++)
	{
		float which_layer = _massList[a]._layer_idx;
		A2DVector moveVector2D = dirVector * (xyGap * which_layer);
		_massList[a]._pos._x += moveVector2D.x;
		_massList[a]._pos._y += moveVector2D.y;
		_massList[a]._pos._z = -(zGap * which_layer);
	}

	if(lockEnds)
	{
		CreateDockPoint(startPt2D + A2DVector(-40, -40), startPt2D, 0);
		CreateDockPoint(endPt2D + A2DVector(40, 40), endPt2D, (SystemParams::_num_layer - 1));
	}

	// flag
	_predefined_time_path = true;

	ResetSpringRestLengths();	
}

//void AnElement::AdjustEnds(A2DVector startPt2D, A2DVector endPt2D, bool lockEnds)
//{
//
//	A3DVector startPt(startPt2D.x, startPt2D.y, 0);
//	A3DVector endPt(endPt2D.x, endPt2D.y, 0);
//	A3DVector dirVector = startPt.DirectionTo(endPt).Norm();
//	float ln = startPt.Distance(endPt);
//	float gapCounter = ln / (float)(SystemParams::_num_layer - 1);
//	
//	for (int a = 0; a < _massList.size(); a++)
//	{
//		int which_layer = _massList[a]._layer_idx;
//		A3DVector moveVector = dirVector * (gapCounter * which_layer);
//		_massList[a]._pos += moveVector;
//
//		if (which_layer == 0 || which_layer == (SystemParams::_num_layer - 1))
//		{
//			_massList[a]._isDocked = lockEnds;
//			_massList[a]._dockPoint = _massList[a]._pos;
//		}
//	}
//
//	ResetSpringRestLengths();
//}

void AnElement::DrawEdges()
{

}

void AnElement::DrawRandomPoints(std::vector<A2DVector> randomPoints)
{
	float imgScale = 2.0f;
	CVImg img;
	img.CreateColorImage(SystemParams::_upscaleFactor * imgScale);
	img.SetColorImageToWhite();

	for(int a = 0; a < randomPoints.size(); a++)
	{
		OpenCVWrapper cvWrapper;
		cvWrapper.DrawCircle(img._img, randomPoints[a] * imgScale, MyColor(255, 0, 0), imgScale);
	}

	std::stringstream ss;
	ss << SystemParams::_save_folder << "PNG\\random_points_" << std::to_string(_elem_idx) << ".png";
	img.SaveImage(ss.str());


}

float AnElement::GetMaxDistRandomPoints(const std::vector<A2DVector>& randomPoints)
{
	NANOFLANNWrapper2D nn;
	nn.SetPointData(randomPoints);
	nn.CreatePointKDTree();

	float maxDist = 0;
	for (int a = 0; a < randomPoints.size(); a++)
	{
		A2DVector pt1 = randomPoints[a];
		std::vector<A2DVector> pts = nn.GetClosestPoints(pt1, 2);
		float d1 = pt1.Distance(pts[0]);
		float d2 = pt1.Distance(pts[1]);

		float d = d2;
		if (d1 > d2) { d = d1; }

		if (d > maxDist) { maxDist = d; }
	}

	std::cout << "max dist = " << maxDist << "\n";

	return maxDist;
}

bool AnElement::IsInsideApprox(int layer_idx, A3DVector pos)
{
	for (unsigned int a = 0; a < _numBoundaryPointPerLayer; a++)
	{
		_a_layer_boundary[a] = _per_layer_boundary[layer_idx][a].GetA2DVector();
	}

	return UtilityFunctions::InsidePolygon(_a_layer_boundary, pos._x, pos._y);
}

//#pragma optimize("", off)
bool AnElement::IsInside(int layer_idx, A3DVector pos, std::vector<A3DVector>& boundary_slice)
{
	int next_layer_idx = layer_idx + 1; // guaranteed exists

	float z_1 = _per_layer_boundary[layer_idx][0]._z;     // negative
	float z_2 = _per_layer_boundary[next_layer_idx][0]._z; // negative
	float interVal = (pos._z - z_1) / 
		             (z_2    - z_1);

	if (interVal > 1 || interVal < 0)
	{
		return false;
	}
	else if(interVal > 1e-5)
	{
		A3DVector pt1, pt1_next, dir1, dir1_unit;
		float dir1_len;

		//std::cout << interVal << "\n";

		for (unsigned int a = 0; a < _numBoundaryPointPerLayer; a++)
		{
			pt1 = _per_layer_boundary[layer_idx][a];
			pt1_next = _per_layer_boundary[next_layer_idx][a];
			dir1 = pt1.DirectionTo(pt1_next);
			//dir1_unit;
			//dir1_len;
			dir1.GetUnitAndDist(dir1_unit, dir1_len);
			_a_layer_boundary_3d[a] = (pt1 + (dir1 * interVal)); // 3D for visualization
			_a_layer_boundary[a] = _a_layer_boundary_3d[a].GetA2DVector(); // 2D for actual test
		}
	}
	else
	{
		_a_layer_boundary_3d = _per_layer_boundary[layer_idx]; // 3D for visualization
		for (unsigned int a = 0; a < _numBoundaryPointPerLayer; a++)
		{
			_a_layer_boundary[a] = _a_layer_boundary_3d[a].GetA2DVector(); // 2D for actual test
		}
	}

	boundary_slice = _a_layer_boundary_3d;
	return UtilityFunctions::InsidePolygon(_a_layer_boundary, pos._x, pos._y);
}

void AnElement::ComputeBary()
{
	// calculate triangles
	std::vector<std::vector<A2DVector>> actualTriangles;
	for (unsigned int c = 0; c < _numTrianglePerLayer; c++) // first layer only
	{
		std::vector<A2DVector> tri(3);
		tri[0] = _massList[_triangles[c].idx0]._pos.GetA2DVector();
		tri[1] = _massList[_triangles[c].idx1]._pos.GetA2DVector();
		tri[2] = _massList[_triangles[c].idx2]._pos.GetA2DVector();
		actualTriangles.push_back(tri);
	}

	//  ================================================  
	// arts
	_arts2Triangles.clear();
	_baryCoords.clear();
	for (unsigned int a = 0; a < _arts.size(); a++)
	{
		std::vector<int> a2t;
		std::vector<ABary> bCoords;
		for (unsigned int b = 0; b < _arts[a].size(); b++)
		{
			int triIdx = -1;
			ABary bary;
			for (unsigned int c = 0; c < _numTrianglePerLayer; c++)  // first layer only
			{
				if (UtilityFunctions::InsidePolygon(actualTriangles[c], _arts[a][b].x, _arts[a][b].y))
				{
					triIdx = c;
					break;
				}
			}

			if (triIdx == -1)
			{
				//std::cout << "art error !!!\n";

				triIdx = -1;
				float dist = 100000000;
				for (unsigned int c = 0; c < _numTrianglePerLayer; c++)  // first layer only
				{
					float d = UtilityFunctions::DistanceToClosedCurve(actualTriangles[c], _arts[a][b]);
					if (d < dist)
					{
						dist = d;
						triIdx = c;
					}
				}
			}

			//else
			{
				bary = UtilityFunctions::Barycentric(_arts[a][b],
					actualTriangles[triIdx][0],
					actualTriangles[triIdx][1],
					actualTriangles[triIdx][2]);
			}
			bCoords.push_back(bary);
			a2t.push_back(triIdx);
		}
		_baryCoords.push_back(bCoords);
		_arts2Triangles.push_back(a2t);
	}
}

void AnElement::Triangularization(std::vector<std::vector<A2DVector>> art_path, int self_idx)
{
	// TODO calculate uniArt
	std::vector<A2DVector> element_path = art_path[0];

	// ----- element index -----
	this->_elem_idx = self_idx;
	
	// -----  why do we need bounding box? ----- 
	A2DRectangle bb = UtilityFunctions::GetBoundingBox(element_path);
	float img_length = bb.witdh;
	if (bb.height > bb.witdh) { img_length = bb.height; }
	A2DVector centerPt = bb.GetCenter();
	//_layer_center = centerPt; // wrong !!!

	// -----  moving to new center ----- 
	img_length += 5.0f; // triangulation error without this ?
	A2DVector newCenter = A2DVector((img_length / 2.0f), (img_length / 2.0f));
	element_path = UtilityFunctions::MovePoly(element_path, centerPt, newCenter);
	for (int a = 0; a < art_path.size(); a++)                                               // moveee
			{ art_path[a] = UtilityFunctions::MovePoly(art_path[a], centerPt, newCenter); } // moveee
	_arts = art_path;

	_layer_center = newCenter;

	// -----  random points ----- 
	std::vector<A2DVector> randomPoints;
	CreateRandomPoints(element_path, img_length, randomPoints, this->_numBoundaryPointPerLayer);
	this->_numPointPerLayer = randomPoints.size(); // ASSIGN

	
	// -----  triangulation ----- 
	OpenCVWrapper cvWrapper;
	std::vector<AnIdxTriangle> tempTriangles;
	std::vector<AnIndexedLine> temp_negSpaceEdges;
	cvWrapper.Triangulate(tempTriangles, temp_negSpaceEdges, randomPoints, element_path, img_length);
	_numTrianglePerLayer = tempTriangles.size(); // number of triangle per layer
	// duplicate triangles
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		float massIdxOffset = a * _numPointPerLayer;
		for (unsigned int b = 0; b < tempTriangles.size(); b++)
		{
			int idx0 = tempTriangles[b].idx0 + massIdxOffset;
			int idx1 = tempTriangles[b].idx1 + massIdxOffset;
			int idx2 = tempTriangles[b].idx2 + massIdxOffset;
			AnIdxTriangle tri(idx0, idx1, idx2);
			_triangles.push_back(tri);
		}
	}
	// duplicate neg space edge
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		float massIdxOffset = a * _numPointPerLayer;
		for (unsigned int b = 0; b < temp_negSpaceEdges.size(); b++)
		{
			int idx0 = temp_negSpaceEdges[b]._index0 + massIdxOffset;
			int idx1 = temp_negSpaceEdges[b]._index1 + massIdxOffset;
			_neg_space_springs.push_back(AnIndexedLine(idx0, idx1));
		}
	}
	// -----  triangulation ----- 

	// ----- interpolation triangles -----	
	/*for (int a = 0; a < SystemParams::_interpolation_factor - 1; a++)  // one less layer
	{
		float massIdxOffset = a * _numPointPerLayer;
		for (unsigned int b = 0; b < tempTriangles.size(); b++)
		{
			int idx0 = tempTriangles[b].idx0 + massIdxOffset;
			int idx1 = tempTriangles[b].idx1 + massIdxOffset;
			int idx2 = tempTriangles[b].idx2 + massIdxOffset;
			AnIdxTriangle tri(idx0, idx1, idx2);
			_interp_triangles.push_back(tri);
		}
	}*/
	// ----- interpolation triangles -----
	

	// z axis offset
	float zOffset = -((float)SystemParams::_upscaleFactor) / ((float)SystemParams::_num_layer - 1);

	// -----  generate mass ----- 
	int massCounter = 0; // self_idx
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		float zPos = zOffset * a;
		for (int b = 0; b < randomPoints.size(); b++)
		{
			AMass m(randomPoints[b].x, // x
				randomPoints[b].y,     // y
				zPos,                  // z
				massCounter++,         // self_idx
				_elem_idx,             // parent_idx
				a);                    // layer_idx
			if (b < _numBoundaryPointPerLayer) { m._is_boundary = true; }
			_massList.push_back(m);                                     
		}
	}
	// ----- generate mass ----- 

	// -----  generate interpolation mass ----- 
	/*int interp_massCounter = 0; // self_idx
	float interp_zOffset = zOffset / ((float)SystemParams::_interpolation_factor );
	for (int a = 0; a < SystemParams::_interpolation_factor - 1; a++) // one less layer
	{
		float zPos = interp_zOffset * (a + 1);
		for (int b = 0; b < randomPoints.size(); b++)
		{
			AMass m(randomPoints[b].x,     // x
					randomPoints[b].y,     // y
					zPos,                  // z, will be changed
					interp_massCounter++,  // self_idx
					_elem_idx,             // parent_idx
					a);                    // layer_idx
			if (b < _numBoundaryPointPerLayer) { m._is_boundary = true; }
			_interp_massList.push_back(m);
		}
	}*/
	// -----  generate interpolation mass ----- 

	// -----  triangle edge springs ----- 
	for (unsigned int a = 0; a < _triangles.size(); a++)
	{
		int idx0 = _triangles[a].idx0;
		int idx1 = _triangles[a].idx1;
		int idx2 = _triangles[a].idx2;

		TryToAddSpring(AnIndexedLine(idx0, idx1), a, _layer_springs, _edgeToTri); // 0 - 1		
		TryToAddSpring(AnIndexedLine(idx1, idx2), a, _layer_springs, _edgeToTri); // 1 - 2		
		TryToAddSpring(AnIndexedLine(idx2, idx0), a, _layer_springs, _edgeToTri); // 2 - 0

		// ----- add triangles to mass -----
		AnIdxTriangle tri(idx0, idx1, idx2);
		_massList[idx0]._triangles.push_back(tri);
		_massList[idx1]._triangles.push_back(tri);
		_massList[idx2]._triangles.push_back(tri);
	}

	// ----- interpolation triangle edge springs ----- 
	/*for (unsigned int a = 0; a < _interp_triangles.size(); a++)
	{
		int idx0 = _interp_triangles[a].idx0;
		int idx1 = _interp_triangles[a].idx1;
		int idx2 = _interp_triangles[a].idx2;

		TryToAddTriangleEdge(AnIndexedLine(idx0, idx1), a, _interp_triEdges, _interp_edgeToTri); // 0 - 1		
		TryToAddTriangleEdge(AnIndexedLine(idx1, idx2), a, _interp_triEdges, _interp_edgeToTri); // 1 - 2		
		TryToAddTriangleEdge(AnIndexedLine(idx2, idx0), a, _interp_triEdges, _interp_edgeToTri); // 2 - 0

		// ----- add triangles to mass -----
		AnIdxTriangle tri(idx0, idx1, idx2);
		_interp_massList[idx0]._triangles.push_back(tri);
		_interp_massList[idx1]._triangles.push_back(tri);
		_interp_massList[idx2]._triangles.push_back(tri);
	}*/

	// ----- bending springs ----- 
	_auxiliary_springs = CreateBendingSprings(_massList, _triangles, _layer_springs, _edgeToTri);
	// from now on _edgeToTri isn't useful anymore
	// ----- bending springs ----- 

	// ----- interpolation bending springs ----- 
	//_interp_auxiliaryEdges = CreateBendingSprings(_interp_massList, _interp_triangles, _interp_triEdges, _interp_edgeToTri);
	// ----- interpolation bending springs ----- 

	//std::cout << "interp tri edge after = " << _interp_triEdges.size() << "\n\n";

	// ----- time triangles -----
	//
	//
	// ----- cur_2 ---- next_2     --> layer t+1
	//        |*         |
	//        | *        |
	//        |  *       |
	//        |   *      |
	//        |    *     |
	//        |     *    |
	//        |      *   |
	//        |       *  |
	//        |        * |
	//        |         *|
	// ----- cur_1 ---- next_1     --> layer t
	int mass_sz = _massList.size();
	for (int a = 0; a < SystemParams::_num_layer - 1; a++)
	{
		int massIdxOffset1 = a * _numPointPerLayer;
		int massIdxOffset2 = massIdxOffset1 + _numPointPerLayer;
		for (int b = 0; b < _numBoundaryPointPerLayer; b++)
		{
			int cur_1 = b + massIdxOffset1; // layer t
			int cur_2 = b + massIdxOffset2; // layer t + 1

			int next_1 = b + 1 + massIdxOffset1; // layer t
			int next_2 = b + 1 + massIdxOffset2; // layer t + 1

			// BE CAREFUL!!!!
			if (b == _numBoundaryPointPerLayer - 1)
			{
				next_1 = massIdxOffset1; 
				next_2 = massIdxOffset2; 
			}

			// BUG !!!
			/*if (cur_1 >= mass_sz ||
				cur_2 >= mass_sz ||
				next_1 >= mass_sz ||
				next_2 >= mass_sz) { continue; }*/

			// layer_idx --> you want to know which layer a triangle belongs to
			int layer_idx = min_of(_massList[cur_1]._layer_idx, 
				                   _massList[cur_2]._layer_idx, 
				                   _massList[next_1]._layer_idx, 
				                   _massList[next_2]._layer_idx);
						
			// cur_1 next_1 cur_2
			AnIdxTriangle tri1(cur_1, next_1, cur_2, layer_idx);
			_surfaceTriangles.push_back(tri1);
			//_massList[cur_1]._timeTriangles.push_back(tri1);
			//_massList[next_1]._timeTriangles.push_back(tri1);
			//_massList[cur_2]._timeTriangles.push_back(tri1);

			// next_1 next_2 cur_2
			AnIdxTriangle tri2(next_1, next_2, cur_2, layer_idx);
			_surfaceTriangles.push_back(tri2);
			//_massList[next_1]._timeTriangles.push_back(tri2);
			//_massList[next_2]._timeTriangles.push_back(tri2);
			//_massList[cur_2]._timeTriangles.push_back(tri2);
		}
	}
	std::cout << "_surfaceTriangles size = " << _surfaceTriangles.size() << "\n";
	// ----- time triangles -----

	// 1-1 pattern
	/*for (int a = 0; a < SystemParams::_num_layer - 1; a++)
	{
		int massIdxOffset1 = a * randomPoints.size();
		int massIdxOffset2 = massIdxOffset1 + randomPoints.size();
		for (int b = 0; b < randomPoints.size(); b++)
		{
			TryToAddTriangleEdge(AnIndexedLine(b + massIdxOffset1, b + massIdxOffset2, true));
		}
	}*/
	//-----------------------
	// cross-straight pattern
	//-----------------------
	for (int a = 0; a < SystemParams::_num_layer - 1; a++)
	{
		int massIdxOffset1 = a * randomPoints.size();
		int massIdxOffset2 = massIdxOffset1 + randomPoints.size();
		for (int b = 0; b < _numBoundaryPointPerLayer; b++)
		{
			int idxA = b - 1; // prev
			int idxB = b + 1; // next
			if (b == 0)
			{
				idxA = _numBoundaryPointPerLayer - 1;
			}
			else if (b == _numBoundaryPointPerLayer - 1)
			{
				idxB = 0;
			}

			// straight
			AddSpring(AnIndexedLine(b + massIdxOffset1, b + massIdxOffset2),    _time_springs); // previously _triEdges

			// cross
			AddSpring(AnIndexedLine(b + massIdxOffset1, idxA + massIdxOffset2), _time_springs); // previously _triEdges

			// cross
			AddSpring(AnIndexedLine(b + massIdxOffset1, idxB + massIdxOffset2), _time_springs); // previously _triEdges

		}
	}
	// -----  triangle edge springs ----- 

	// ----- interpolation cross pattern -----
	// ----- FIRST -----
	/*for (int a = 0; a < _numBoundaryPointPerLayer; a++)
	{
		int idxA = a - 1; // prev
		int idxB = a + 1; // next
		if (a == 0)
		{
			idxA = _numBoundaryPointPerLayer - 1;
		}
		else if (a == _numBoundaryPointPerLayer - 1)
		{
			idxB = 0;
		}

		// first index is from original, second index is from interpolation
		ForceAddTriangleEdge(AnIndexedLine(a, idxA, true), -1, _timeEdgesA, _interp_edgeToTriA);
		ForceAddTriangleEdge(AnIndexedLine(a, idxB, true), -1, _timeEdgesA, _interp_edgeToTriA);
	}*/

	// ----- MID -----
	/*for (int a = 0; a < SystemParams::_interpolation_factor - 2; a++) // two less
	{
		int massIdxOffset1 = a * _numPointPerLayer;
		int massIdxOffset2 = massIdxOffset1 + _numPointPerLayer;
		for (int b = 0; b < _numBoundaryPointPerLayer; b++)
		{
			int idxA = b - 1; // prev
			int idxB = b + 1; // next
			if (b == 0)
			{
				idxA = _numBoundaryPointPerLayer - 1;
			}
			else if (b == _numBoundaryPointPerLayer - 1)
			{
				idxB = 0;
			}

			TryToAddTriangleEdge(AnIndexedLine(b + massIdxOffset1, idxA + massIdxOffset2, true), -1, _interp_triEdges, _interp_edgeToTri);
			TryToAddTriangleEdge(AnIndexedLine(b + massIdxOffset1, idxB + massIdxOffset2, true), -1, _interp_triEdges, _interp_edgeToTri);
		}
	}*/
	// ----- END -----
	/*int interp_factor = (SystemParams::_interpolation_factor - 2) * _numPointPerLayer;
	int ori_factor = _numPointPerLayer;
	for (int a = 0; a < _numBoundaryPointPerLayer; a++)
	{
		int idxA = a - 1; // prev
		int idxB = a + 1; // next
		if (a == 0)
		{
			idxA = _numBoundaryPointPerLayer - 1;
		}
		else if (a == _numBoundaryPointPerLayer - 1)
		{
			idxB = 0;
		}

		// first index is from interpolation, second index is from original
		ForceAddTriangleEdge(AnIndexedLine(interp_factor + a, ori_factor + idxA, true), -1, _timeEdgesB, _interp_edgeToTriB);
		ForceAddTriangleEdge(AnIndexedLine(interp_factor + a, ori_factor + idxB, true), -1, _timeEdgesB, _interp_edgeToTriB);
	}*/

	// rotate
	CreateHelix();

	// reset !!!
	ResetSpringRestLengths();
	//Interp_ResetSpringRestLengths();
	

	// ----- some precomputation ----- 
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		_per_layer_boundary.push_back(std::vector<A3DVector>());
		//_per_layer_boundary_drawing.push_back(std::vector<A3DVector>());
	}
	for (int a = 0; a < _massList.size(); a++)
	{
		int perLayerIdx = a % _numPointPerLayer;
		if (perLayerIdx < _numBoundaryPointPerLayer)
		{
			int layerIdx = _massList[a]._layer_idx;
			_per_layer_boundary[layerIdx].push_back ( _massList[a]._pos );
			//_per_layer_boundary_drawing[layerIdx].push_back(_massList[a]._pos);
		}
	}

	// for closest point
	for (int a = 0; a < _numBoundaryPointPerLayer; a++)
	{
		_a_layer_boundary.push_back(A2DVector());
		_a_layer_boundary_3d.push_back(A3DVector());
	}

	// ----- some precomputation or interpolation ----- 
	/*for (int a = 0; a < SystemParams::_interpolation_factor - 1; a++)
	{
		_interp_per_layer_boundary.push_back(std::vector<A2DVector>());
	}
	for (int a = 0; a < _interp_massList.size(); a++)
	{
		int perLayerIdx = a % _numPointPerLayer;
		if (perLayerIdx < _numBoundaryPointPerLayer)
		{
			int layerIdx = _interp_massList[a]._layer_idx;
			_interp_per_layer_boundary[layerIdx].push_back(_interp_massList[a]._pos.GetA2DVector());
		}
	}*/
}

void AnElement::CalculateRestStructure()
{
	//UpdateLayerBoundaries(); // update per_layer_boundary

	//std::vector<A2DVector> _layer_center_array; // OpenCVWrapper::GetCenter
	OpenCVWrapper cvWrapper;
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		A2DVector centerPt = cvWrapper.GetCenter(_per_layer_boundary[a]);
		_layer_center_array.push_back(centerPt);
	}

	//std::vector<A3DVector> _rest_mass_pos_array;
	_rest_mass_pos_array.clear();
	for (int a = 0; a < _massList.size(); a++)
	{
		_ori_rest_mass_pos_array.push_back(_massList[a]._pos);
		_rest_mass_pos_array.push_back(_massList[a]._pos);
	}
	
}

void AnElement::Grow(float growth_scale_iter, float dt)
{
	if (_scale > _maxScale)
	{
		return;
	}

	_scale += growth_scale_iter * dt;

	// iterate rest_mass_pos
	for (int a = 0; a < _rest_mass_pos_array.size(); a++)
	{
		int layer_idx = _massList[a]._layer_idx;
		A2DVector pos = _ori_rest_mass_pos_array[a].GetA2DVector();
		pos -= _layer_center_array[layer_idx];		
		pos *= _scale;
		pos += _layer_center_array[layer_idx];
		_rest_mass_pos_array[a]._x = pos.x;
		_rest_mass_pos_array[a]._y = pos.y;
	}

	// iterate edges
	for (unsigned int a = 0; a < _layer_springs.size(); a++)
	{
		A3DVector p1 = _rest_mass_pos_array[_layer_springs[a]._index0];
		A3DVector p2 = _rest_mass_pos_array[_layer_springs[a]._index1];
		_layer_springs[a].SetActualOriDistance(p1.Distance(p2));
	}

	// iterate edges
	for (unsigned int a = 0; a < _time_springs.size(); a++)
	{
		A3DVector p1 = _rest_mass_pos_array[_time_springs[a]._index0];
		A3DVector p2 = _rest_mass_pos_array[_time_springs[a]._index1];
		_time_springs[a].SetActualOriDistance(p1.Distance(p2));
	}

	for (unsigned int a = 0; a < _auxiliary_springs.size(); a++)
	{
		A3DVector p1 = _rest_mass_pos_array[_auxiliary_springs[a]._index0];
		A3DVector p2 = _rest_mass_pos_array[_auxiliary_springs[a]._index1];
		_auxiliary_springs[a].SetActualOriDistance(p1.Distance(p2));
	}

	for (unsigned int a = 0; a < _neg_space_springs.size(); a++)
	{
		A3DVector p1 = _rest_mass_pos_array[_neg_space_springs[a]._index0];
		A3DVector p2 = _rest_mass_pos_array[_neg_space_springs[a]._index1];
		_neg_space_springs[a].SetActualOriDistance(p1.Distance(p2));
	}
}

void AnElement::CreateRandomPoints(std::vector<A2DVector> ornamentBoundary,
							float img_length,
							std::vector<A2DVector>& randomPoints,
							int& boundaryPointNum)
{
	// how many points (really weird code...)
	float fVal = img_length / SystemParams::_upscaleFactor;
	fVal *= fVal;
	int numPoints = SystemParams::_sampling_density * fVal;
	float resamplingGap = std::sqrt(float(numPoints)) / float(numPoints) * img_length;

	std::vector<A2DVector> resampledBoundary;

	ornamentBoundary.push_back(ornamentBoundary[0]); // closed sampling
	float rGap = (float)(resamplingGap * SystemParams::_boundary_sampling_factor);
	UtilityFunctions::UniformResample(ornamentBoundary, resampledBoundary, rGap);
	// bug !!! nasty code
	if (resampledBoundary[resampledBoundary.size() - 1].Distance(resampledBoundary[0]) < rGap * 0.5) // r gap
	{
		resampledBoundary.pop_back();
	}
	
	DefaultPRNG PRNG;
	if (SystemParams::_seed > 0)
	{
		PRNG = DefaultPRNG(SystemParams::_seed);
	}
	PoissonGenerator pg;
	const auto points = pg.GeneratePoissonPoints(numPoints, PRNG);

	randomPoints.insert(randomPoints.begin(), resampledBoundary.begin(), resampledBoundary.end());
	boundaryPointNum = resampledBoundary.size();

	float sc = img_length * std::sqrt(2.0f);
	float ofVal = 0.5f * (sc - img_length);
	//float ofVal = 0;
	// ---------- iterate points ----------
	for (auto i = points.begin(); i != points.end(); i++)
	{
		float x = (i->x * sc) - ofVal;
		float y = (i->y * sc) - ofVal;
		A2DVector pt(x, y);

		if (UtilityFunctions::InsidePolygon(ornamentBoundary, pt.x, pt.y))
		{
			float d = UtilityFunctions::DistanceToClosedCurve(resampledBoundary, pt);
			if (d > resamplingGap)
			{
				randomPoints.push_back(pt);
			}
			//AVector cPt = knn->GetClosestPoints(pt, 1)[0];
			//if (cPt.Distance(pt) > resamplingGap)
			//	{ randomPoints.push_back(pt); }
		}
	}
}

// visualization
void AnElement::InitMeshOgre3D(Ogre::SceneManager* sceneMgr,
								Ogre::SceneNode* sceneNode,
								const Ogre::String& name,
								const Ogre::String& materialName)
{
	this->_sceneMgr = sceneMgr;
	this->_sceneNode = sceneNode;

	if (_tubeObject) return;

	float rVal = (float)(rand() % 255) / 255.0f;
	float gVal = (float)(rand() % 255) / 255.0f;
	float bVal = (float)(rand() % 255) / 255.0f;

	// Coloe of this element, very important
	this->_color = MyColor(rVal * 255, gVal * 255, bVal * 255);

	// ---------- neg_space ----------
	Ogre::MaterialPtr neg_sp_line_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("neg_sp_line_material" + std::to_string(_elem_idx));
	neg_sp_line_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(rVal, gVal, bVal, 1));
	_neg_space_edge_lines = new DynamicLines(neg_sp_line_material, Ogre::RenderOperation::OT_LINE_LIST);
	for (int l = 0; l < _neg_space_springs.size(); l++)
	{
		A3DVector pt1 = _massList[_neg_space_springs[l]._index0]._pos;
		A3DVector pt2 = _massList[_neg_space_springs[l]._index1]._pos;
		_neg_space_edge_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
		_neg_space_edge_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
	}
	_neg_space_edge_lines->update();
	_neg_space_edge_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("_neg_space_edge_lines_" + std::to_string(_elem_idx));
	_neg_space_edge_node->attachObject(_neg_space_edge_lines);

	
	// ---------- force ----------
	Ogre::MaterialPtr force_line_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("force_line_material_" + std::to_string(_elem_idx));
	force_line_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));
	_force_lines = new DynamicLines(force_line_material, Ogre::RenderOperation::OT_LINE_LIST);
	for (int a = 0; a < _massList.size(); a++)
	{
		A3DVector pos1 = _massList[a]._pos;
		A3DVector pos2 = pos1 + A3DVector(2, 0, 0);
		_force_lines->addPoint(pos1._x, pos1._y, pos1._z);
		_force_lines->addPoint(pos2._x, pos2._y, pos2._z);
	}
	_force_lines->update();
	_force_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("_force_node_" + std::to_string(_elem_idx));
	_force_node->attachObject(_force_lines);

	// ---------- material ----------
	Ogre::MaterialPtr line_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("ElementLines" + std::to_string(_elem_idx));	
	line_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(rVal, gVal, bVal, 1));
	
	// ---------- mass list ----------
	_massList_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
	{
		float offsetVal = 2;
		for (int a = 0; a < _massList.size(); a++)
		{
			_massList_lines->addPoint(_massList[a]._pos._x - offsetVal, _massList[a]._pos._y, _massList[a]._pos._z);
			_massList_lines->addPoint(_massList[a]._pos._x + offsetVal, _massList[a]._pos._y, _massList[a]._pos._z);
			_massList_lines->addPoint(_massList[a]._pos._x, _massList[a]._pos._y - offsetVal, _massList[a]._pos._z);
			_massList_lines->addPoint(_massList[a]._pos._x, _massList[a]._pos._y + offsetVal, _massList[a]._pos._z);
		}
	}
	_massList_lines->update();
	_massList_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("_massList_lines_" + std::to_string(_elem_idx));
	_massList_node->attachObject(_massList_lines);

	// --------- time edges ----------
	_time_edge_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
	for (int a = 0; a < _time_springs.size(); a++)
	{
		//if(_triEdges[a]._isLayer2Layer)
		//{
		A3DVector pos1 = _massList[_time_springs[a]._index0]._pos;
		A3DVector pos2 = _massList[_time_springs[a]._index1]._pos;
		_time_edge_lines->addPoint(Ogre::Vector3(pos1._x, pos1._y, pos1._z));
		_time_edge_lines->addPoint(Ogre::Vector3(pos2._x, pos2._y, pos2._z));
		//}
	}
	_time_edge_lines->update();
	_time_edge_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("time_edge_node_" + std::to_string(_elem_idx));
	_time_edge_node->attachObject(_time_edge_lines);

	// ---------- boundary ----------
	_boundary_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
	for (int l = 0; l < SystemParams::_num_layer; l++)
	{
		int layerOffset = l * _numPointPerLayer;
		for (int b = 0; b < _numBoundaryPointPerLayer; b++)
		{
			int massIdx1 = b + layerOffset;
			int massIdx2 = b + layerOffset + 1;
			if (b == _numBoundaryPointPerLayer - 1)
			{
				massIdx2 = layerOffset;
			}
			A3DVector pt1 = _massList[massIdx1]._pos;
			A3DVector pt2 = _massList[massIdx2]._pos;
			_boundary_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_boundary_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
	}	
	_boundary_lines->update();
	_boundary_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("debug_lines_" + std::to_string(_elem_idx));
	_boundary_node->attachObject(_boundary_lines);

	// ---------- time triangles ----------
	_surface_tri_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
	std::vector<A3DVector> tri(3);
	for (int b = 0; b < _surfaceTriangles.size(); b++)
	{
		tri[0] = _massList[_surfaceTriangles[b].idx0]._pos;
		tri[1] = _massList[_surfaceTriangles[b].idx1]._pos;
		tri[2] = _massList[_surfaceTriangles[b].idx2]._pos;

		_surface_tri_lines->addPoint(Ogre::Vector3(tri[0]._x, tri[0]._y, tri[0]._z));
		_surface_tri_lines->addPoint(Ogre::Vector3(tri[1]._x, tri[1]._y, tri[1]._z));

		_surface_tri_lines->addPoint(Ogre::Vector3(tri[1]._x, tri[1]._y, tri[1]._z));
		_surface_tri_lines->addPoint(Ogre::Vector3(tri[2]._x, tri[2]._y, tri[2]._z));

		_surface_tri_lines->addPoint(Ogre::Vector3(tri[2]._x, tri[2]._y, tri[2]._z));
		_surface_tri_lines->addPoint(Ogre::Vector3(tri[0]._x, tri[0]._y, tri[0]._z));

	}
	_surface_tri_lines->update();
	_surface_tri_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("time_tri_node_" + std::to_string(_elem_idx));
	_surface_tri_node->attachObject(_surface_tri_lines);

	// ---------- closest point approx debug  BACK ----------
	/*Ogre::MaterialPtr line_material_c_pt_approx_back = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("ClosestPtApproxMatback_" + std::to_string(_elem_idx));
	line_material_c_pt_approx_back->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(0, 0, 1, 1));
	_closet_pt_approx_lines_back = new DynamicLines(line_material_c_pt_approx_back, Ogre::RenderOperation::OT_LINE_LIST);
	_closet_pt_approx_lines_back->update();
	_closet_pt_approx_node_back = _sceneMgr->getRootSceneNode()->createChildSceneNode("closest_point_approx_lines_back_" + std::to_string(_elem_idx));
	_closet_pt_approx_node_back->attachObject(_closet_pt_approx_lines_back);*/


	// ---------- closest point debug BACK ----------
	/*
	Ogre::MaterialPtr line_material_c_pt_back = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("ClosestPtMatback_" + std::to_string(_elem_idx));
	line_material_c_pt_back->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));
	_closet_pt_lines_back = new DynamicLines(line_material_c_pt_back, Ogre::RenderOperation::OT_LINE_LIST);
	_closet_pt_lines_back->update();
	_closet_pt_node_back = _sceneMgr->getRootSceneNode()->createChildSceneNode("closest_point_debug_lines_back_" + std::to_string(_elem_idx));
	_closet_pt_node_back->attachObject(_closet_pt_lines_back);
	*/
	// ---------- closest point approx debug ----------
	Ogre::MaterialPtr line_material_c_pt_approx = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("ClosestPtApproxMat" + std::to_string(_elem_idx));
	line_material_c_pt_approx->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));
	_closet_pt_approx_lines = new DynamicLines(line_material_c_pt_approx, Ogre::RenderOperation::OT_LINE_LIST);
	_closet_pt_approx_lines->update();
	_closet_pt_approx_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("closest_point_approx_lines_" + std::to_string(_elem_idx));
	_closet_pt_approx_node->attachObject(_closet_pt_approx_lines);


	// ---------- closest point debug ----------
	Ogre::MaterialPtr line_material_c_pt = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("Closest_Pt_Mat_" + std::to_string(_elem_idx));
	line_material_c_pt->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(0, 0, 1, 1));
	_closet_pt_lines = new DynamicLines(line_material_c_pt, Ogre::RenderOperation::OT_LINE_LIST);
	_closet_pt_lines->update();
	_closet_pt_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("closest_point_debug_lines_" + std::to_string(_elem_idx));
	_closet_pt_node->attachObject(_closet_pt_lines);

	// ---------- dock debug ----------
	if (_dock_mass_idx.size() > 0)
	{
		Ogre::MaterialPtr line_material_2 = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("DockLines" + std::to_string(_elem_idx));
		line_material_2->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));
		_dock_lines = new DynamicLines(line_material_2, Ogre::RenderOperation::OT_LINE_LIST);

		for (int a = 0; a < _dock_mass_idx.size(); a++)
		{
			int massIdx = _dock_mass_idx[a];
			_dock_lines->addPoint(Ogre::Vector3(_massList[massIdx]._pos._x, _massList[massIdx]._pos._y, _massList[massIdx]._pos._z));
			_dock_lines->addPoint(Ogre::Vector3(_massList[massIdx]._dockPoint._x, _massList[massIdx]._dockPoint._y, _massList[massIdx]._dockPoint._z));
		}

		_dock_lines->update();
		_dock_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("debug_lines_2_" + std::to_string(_elem_idx));
		_dock_node->attachObject(_dock_lines);
	}	


	// ---------- debug closest surface tri ----------
	Ogre::MaterialPtr line_material_ctri = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("line_material_ctri" + std::to_string(_elem_idx));
	line_material_ctri->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));

	_closest_tri_lines = new DynamicLines(line_material_ctri, Ogre::RenderOperation::OT_LINE_LIST);
	_closest_tri_lines->update();
	_closest_tri_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("_closest_tri_lines_" + std::to_string(_elem_idx));
	_closest_tri_node->attachObject(_closest_tri_lines);
	   
	// ---------- debug closest slice ----------
	Ogre::MaterialPtr line_material_asdf = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("line_material_asdf_" + std::to_string(_elem_idx));
	line_material_asdf->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));

	

	_closest_slice_lines = new DynamicLines(line_material_asdf, Ogre::RenderOperation::OT_LINE_LIST);
	_closest_slice_lines->update();
	_closest_slice_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("_closest_slice_lines_" + std::to_string(_elem_idx));
	_closest_slice_node->attachObject(_closest_slice_lines);

	// ---------- Overlap debug! ----------
	Ogre::MaterialPtr line_material_ovlp = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("line_material_ovlp" + std::to_string(_elem_idx));
	line_material_ovlp->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(0, 0, 0, 1));
	_overlap_lines = new DynamicLines(line_material_ovlp, Ogre::RenderOperation::OT_LINE_LIST);
	_overlap_lines->update();
	_overlap_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("_overlap_lines_" + std::to_string(_elem_idx));
	_overlap_node->attachObject(_overlap_lines);

	// check DynamicLines.cpp
	// mBox.setExtents(Vector3(-50000, -50000, -50000), Vector3(50000, 50000, 50000));

	/*_neg_space_edge_lines->setBoundingBox(Ogre::AxisAlignedBox(0, 0, 0, 1, 1, 1));
	_boundary_lines->setBoundingBox(Ogre::AxisAlignedBox(0, 0, 0, 1, 1, 1));
	_surface_tri_lines->setBoundingBox(Ogre::AxisAlignedBox(0, 0, 0, 1, 1, 1));
	_closet_pt_approx_lines_back->setBoundingBox(Ogre::AxisAlignedBox(0, 0, 0, 1, 1, 1));
	_closet_pt_lines_back->setBoundingBox(Ogre::AxisAlignedBox(0, 0, 0, 1, 1, 1));
	_closet_pt_approx_lines->setBoundingBox(Ogre::AxisAlignedBox(0, 0, 0, 1, 1, 1));
	_closet_pt_lines->setBoundingBox(Ogre::AxisAlignedBox(0, 0, 0, 1, 1, 1));
	_closest_slice_lines->setBoundingBox(Ogre::AxisAlignedBox(0, 0, 0, 1, 1, 1));
	_overlap_lines->setBoundingBox(Ogre::AxisAlignedBox(0, 0, 0, 1, 1, 1));*/
}

void AnElement::UpdateClosestTriOgre3D()
{
	if (SystemParams::_show_closest_tri)
	{
		_closest_tri_node->setVisible(true);
		/*_closest_tri_lines->clear();
		std::vector<A3DVector> tri;

		for (unsigned int a = 0; a < _massList.size(); a++)
		{
			if (_massList[a]._is_boundary && _massList[a]._closest_elem_idx != -1)
			{
				if (SystemParams::_layer_slider_int >= 0 && _massList[a]._layer_idx != SystemParams::_layer_slider_int) { continue; }

				for (int b = 0; b < _massList[a]._closest_tri_array.size(); b++)
				{
					tri = _massList[a]._closest_tri_array[b];
					_closest_tri_lines->addPoint(Ogre::Vector3(tri[0]._x, tri[0]._y, tri[0]._z));
					_closest_tri_lines->addPoint(Ogre::Vector3(tri[1]._x, tri[1]._y, tri[1]._z));

					_closest_tri_lines->addPoint(Ogre::Vector3(tri[1]._x, tri[1]._y, tri[1]._z));
					_closest_tri_lines->addPoint(Ogre::Vector3(tri[2]._x, tri[2]._y, tri[2]._z));

					_closest_tri_lines->addPoint(Ogre::Vector3(tri[2]._x, tri[2]._y, tri[2]._z));
					_closest_tri_lines->addPoint(Ogre::Vector3(tri[0]._x, tri[0]._y, tri[0]._z));
				}
				
			}
		}*/
	}
	else
	{
		_closest_tri_node->setVisible(false);
	}

	_closest_tri_lines->update();
}

void AnElement::UpdateClosestSliceOgre3D()
{
	if(SystemParams::_show_overlap)
	{
		_closest_slice_node->setVisible(true);
		_closest_slice_lines->clear();

		for (unsigned int a = 0; a < _massList.size(); a++)
		{
			if (_massList[a]._is_boundary && _massList[a]._is_inside)
			{
				std::vector<A3DVector> slice_array = _massList[a]._closest_boundary_slice;

				for (int i = 0; i < slice_array.size(); i++)
				{
					int next_i = i + 1;
					if (i == slice_array.size() - 1)
					{
						next_i = 0;
					}
					_closest_slice_lines->addPoint(Ogre::Vector3(slice_array[i]._x,      slice_array[i]._y,      slice_array[i]._z));
					_closest_slice_lines->addPoint(Ogre::Vector3(slice_array[next_i]._x, slice_array[next_i]._y, slice_array[next_i]._z));
				}

			}
		}

	}
	else
	{
		_closest_slice_node->setVisible(false);
	}

	_closest_slice_lines->update();
}

void AnElement::UpdateDockLinesOgre3D()
{
	if (_dock_mass_idx.size() == 0) { return; }

	for (int a = 0; a < _dock_mass_idx.size(); a++)
	{
		int idx1 = a * 2;
		int idx2 = idx1 + 1;
		int massIdx = _dock_mass_idx[a];
		_dock_lines->setPoint(idx1, Ogre::Vector3(_massList[massIdx]._pos._x, _massList[massIdx]._pos._y, _massList[massIdx]._pos._z));
		_dock_lines->setPoint(idx2, Ogre::Vector3(_massList[massIdx]._dockPoint._x, _massList[massIdx]._dockPoint._y, _massList[massIdx]._dockPoint._z));
	}

	_dock_lines->update();
}

void AnElement::UpdateNegSpaceEdgeOgre3D()
{
	if(SystemParams::_show_negative_space_springs)
	{
		_neg_space_edge_node->setVisible(true);
		int idx = 0;

		for (int l = 0; l < _neg_space_springs.size(); l++)
		{
			A3DVector pt1 = _massList[_neg_space_springs[l]._index0]._pos;
			A3DVector pt2 = _massList[_neg_space_springs[l]._index1]._pos;
			_neg_space_edge_lines->setPoint(idx++, Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_neg_space_edge_lines->setPoint(idx++, Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
		_neg_space_edge_lines->update();
	}
	else
	{
		_neg_space_edge_node->setVisible(false);
	}
}

void AnElement::UpdateBoundaryDisplayOgre3D()
{
	if(SystemParams::_show_element_boundaries)
	{
		_boundary_node->setVisible(true);
		int idx = 0;

		A3DVector pt1;
		A3DVector pt2;

		for (int l = 0; l < SystemParams::_num_layer; l++)
		{
			int layerOffset = l * _numPointPerLayer;
			for (int b = 0; b < _numBoundaryPointPerLayer; b++)
			{
				int massIdx1 = b + layerOffset;
				int massIdx2 = b + layerOffset + 1;
				if (b == _numBoundaryPointPerLayer - 1)
				{
					massIdx2 = layerOffset;
				}

				if(SystemParams::_layer_slider_int == -1)
				{					
					pt1 = _massList[massIdx1]._pos;
					pt2 = _massList[massIdx2]._pos;
				}
				else if (l == SystemParams::_layer_slider_int)
				{
					pt1 = _massList[massIdx1]._pos;
					pt2 = _massList[massIdx2]._pos;
				}
				else
				{
					pt1 = A3DVector(-100, -100, -100);
					pt2 = A3DVector(-100, -100, -100);
					
				}
				_boundary_lines->setPoint(idx++, Ogre::Vector3(pt1._x, pt1._y, pt1._z));
				_boundary_lines->setPoint(idx++, Ogre::Vector3(pt2._x, pt2._y, pt2._z));
			}
		}

		_boundary_lines->update();
	}
	else
	{
		_boundary_node->setVisible(false);
	}
}

void AnElement::ShowTimeSprings(bool yesno)
{
	//_time_springs_node->setVisible(yesno);
}

void AnElement::UpdateSurfaceTriangleOgre3D()
{	
	if(SystemParams::_show_surface_tri)
	{
		_surface_tri_node->setVisible(true);
		int idx = 0;
		std::vector<A3DVector> tri(3);
		for (unsigned int b = 0; b < _surfaceTriangles.size(); b++)
		{
			//std::vector<A3DVector> tri;
			tri[0] = _massList[_surfaceTriangles[b].idx0]._pos;
			tri[1] = _massList[_surfaceTriangles[b].idx1]._pos;
			tri[2] = _massList[_surfaceTriangles[b].idx2]._pos;

			_surface_tri_lines->setPoint(idx++, Ogre::Vector3(tri[0]._x, tri[0]._y, tri[0]._z));
			_surface_tri_lines->setPoint(idx++, Ogre::Vector3(tri[1]._x, tri[1]._y, tri[1]._z));

			_surface_tri_lines->setPoint(idx++, Ogre::Vector3(tri[1]._x, tri[1]._y, tri[1]._z));
			_surface_tri_lines->setPoint(idx++, Ogre::Vector3(tri[2]._x, tri[2]._y, tri[2]._z));

			_surface_tri_lines->setPoint(idx++, Ogre::Vector3(tri[2]._x, tri[2]._y, tri[2]._z));
			_surface_tri_lines->setPoint(idx++, Ogre::Vector3(tri[0]._x, tri[0]._y, tri[0]._z));
		}		
	}
	else
	{
		_surface_tri_node->setVisible(false);
	}

	//_surface_tri_lines->setBoundingBox(Ogre::AxisAlignedBox(5, 5, 5, 6, 6, 6));
	_surface_tri_lines->update();
	
}

void AnElement::UpdateOverlapOgre3D()
{
	if (SystemParams::_show_overlap)
	{
		_overlap_node->setVisible(true);

		float plus_offset = 5;

		_overlap_lines->clear();

		for (int a = 0; a < _massList.size(); a++)
		{
			if (_massList[a]._is_boundary && _massList[a]._is_inside)
			{
				A3DVector massPos = _massList[a]._pos;

				_overlap_lines->addPoint(massPos._x - plus_offset, massPos._y, massPos._z);
				_overlap_lines->addPoint(massPos._x + plus_offset, massPos._y, massPos._z);
				_overlap_lines->addPoint(massPos._x, massPos._y - plus_offset, massPos._z);
				_overlap_lines->addPoint(massPos._x, massPos._y + plus_offset, massPos._z);
			}
		}
	}
	else
	{
		_overlap_node->setVisible(false);
	}

	

	_overlap_lines->update();
}

void AnElement::UpdateForceOgre3D()
{
	if (SystemParams::_show_force)
	{
		_force_node->setVisible(true);

		int idx = 0;
		for (int a = 0; a < _massList.size(); a++)
		{
			A3DVector pos1 = _massList[a]._pos;

			A3DVector vel = _massList[a]._velocity;
			A3DVector norm;
			float dist;
			vel.GetUnitAndDist(norm, dist);

			A3DVector pos2 = pos1 + norm * dist * 20;

			_force_lines->setPoint(idx++, Ogre::Vector3(pos1._x, pos1._y, pos1._z));
			_force_lines->setPoint(idx++, Ogre::Vector3(pos2._x, pos2._y, pos2._z));
		}

	}
	else
	{
		_force_node->setVisible(false);
	}

	
	_force_lines->update();
}

void AnElement::UpdateTimeEdgesOgre3D()
{
	if (SystemParams::_show_time_edges)
	{
		_time_edge_node->setVisible(true);

		int idx = 0;
		for (int a = 0; a < _time_springs.size(); a++)
		{
			//if (_triEdges[a]._isLayer2Layer)
			//{
			A3DVector pos1 = _massList[_time_springs[a]._index0]._pos;
			A3DVector pos2 = _massList[_time_springs[a]._index1]._pos;
			_time_edge_lines->setPoint(idx++, Ogre::Vector3(pos1._x, pos1._y, pos1._z));
			_time_edge_lines->setPoint(idx++, Ogre::Vector3(pos2._x, pos2._y, pos2._z));
			//}
		}
		_time_edge_lines->update();
	}
	else
	{
		_time_edge_node->setVisible(false);
	}

	_time_edge_lines->update();
}

void AnElement::UpdateMassListOgre3D()
{
	if(SystemParams::_show_mass_list)
	{
		_massList_node->setVisible(true);
		int idx = 0;


		float offsetVal = 1;
		for (int a = 0; a < _massList.size(); a++)
		{
			_massList_lines->setPoint(idx++, Ogre::Vector3(_massList[a]._pos._x - offsetVal, _massList[a]._pos._y, _massList[a]._pos._z));
			_massList_lines->setPoint(idx++, Ogre::Vector3(_massList[a]._pos._x + offsetVal, _massList[a]._pos._y, _massList[a]._pos._z));
			_massList_lines->setPoint(idx++, Ogre::Vector3(_massList[a]._pos._x, _massList[a]._pos._y - offsetVal, _massList[a]._pos._z));
			_massList_lines->setPoint(idx++, Ogre::Vector3(_massList[a]._pos._x, _massList[a]._pos._y + offsetVal, _massList[a]._pos._z));
		}
	}
	else
	{
		_massList_node->setVisible(false);
	}

	_massList_lines->update();
}

void AnElement::UpdateSpringDisplayOgre3D()
{
	/*if (!SystemParams::_show_time_tri) { return; }

	int idx = 0;

	for (int a = 0; a < _triEdges.size(); a++)
	{
		AnIndexedLine ln = _triEdges[a];

		//if (ln._isLayer2Layer) { continue; }

		A3DVector pt1 = _massList[ln._index0]._pos;
		A3DVector pt2 = _massList[ln._index1]._pos;
		_spring_lines->setPoint(idx++, Ogre::Vector3(pt1._x, pt1._y, pt1._z));
		_spring_lines->setPoint(idx++, Ogre::Vector3(pt2._x, pt2._y, pt2._z));
	}
	for (int a = 0; a < _auxiliaryEdges.size(); a++)
	{
		AnIndexedLine ln = _auxiliaryEdges[a];


		A3DVector pt1 = _massList[ln._index0]._pos;
		A3DVector pt2 = _massList[ln._index1]._pos;
		_spring_lines->setPoint(idx++, Ogre::Vector3(pt1._x, pt1._y, pt1._z));
		_spring_lines->setPoint(idx++, Ogre::Vector3(pt2._x, pt2._y, pt2._z));
	}

	_spring_lines->update();*/
}


// visualization
void AnElement::UpdateMeshOgre3D()
{
	if (_tubeObject->getDynamic() == true && _tubeObject->getNumSections() > 0)
		_tubeObject->beginUpdate(0);
	else
		_tubeObject->begin(_material->getName());

	for (int a = 0; a < _massList.size(); a++)
	{
		if (a % 11 == 0) continue;

		A3DVector pos = _massList[a]._pos;
		_tubeObject->position(pos._x, pos._y, pos._z);

		// normal doesn't work???
		//A3DVector normVec = A3DVector(250, 250, pos._z).DirectionTo(pos).Norm();
		//_tubeObject->normal(normVec._x, normVec._y, normVec._z);

		// uv
		int curLayer = a / 1;
		int idx = a % 10;
		float u = (float)idx / 10.0;
		float v = (float)curLayer / (float)(5);
		_tubeObject->textureCoord(u, v);
	}

	//std::cout << "_massList size = " << _massList.size() << "\n";
	//std::cout << "render vertex = " << _tubeObject->getCurrentVertexCount() << "\n";

	int indexOffset = 10;
	int A, B, C, D;
	int maxIdx = SystemParams::_num_layer - 1;
	for (int i = 0; i < maxIdx; i++)
	{
		int startIdx = i * indexOffset;
		// 0
		{
			A = startIdx;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 1
		{
			A = startIdx + 1;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 2
		{
			A = startIdx + 2;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 3
		{
			A = startIdx + 3;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 4
		{
			A = startIdx + 4;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 5
		{
			A = startIdx + 5;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 6
		{
			A = startIdx + 6;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 7
		{
			A = startIdx + 7;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 8
		{
			A = startIdx + 8;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 9
		{
			A = startIdx + 9;
			B = startIdx;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}
	}

	_tubeObject->end();

}

/*void AnElement::UpdateSpringLengths()
{
	for (int a = 0; a < _triEdges.size(); a++)
	{
		A3DVector pt1 = _massList[_triEdges[a]._index0]._pos;
		A3DVector pt2 = _massList[_triEdges[a]._index1]._pos;
		float d = pt1.Distance(pt2);
		_triEdges[a]._dist = d;
	}
}*/

void AnElement::RandomizeLayerSize()
{
	float scale1 = (float)((rand() % 100) + 50) / 100.0f;
	float scale2 = (float)((rand() % 100) + 50) / 100.0f;
	float scale3 = (float)((rand() % 100) + 50) / 100.0f;
	float scale4 = (float)((rand() % 100) + 50) / 100.0f;

	int idx2 = SystemParams::_num_layer / 3;
	int idx3 = 2 * SystemParams::_num_layer / 3;

	float athird = SystemParams::_num_layer / 3;


	std::vector<float> randomScale;

	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		float scVal = 1.0f;
		if (a >= 0 && a < idx2)
		{
			float ratioVal = (float)a / athird;
			scVal = (1.0 - ratioVal) * scale1 + ratioVal * scale2;
		}
		else if (a >= idx2 && a < idx3)
		{
			float ratioVal = (float)(a - idx2) / athird;
			scVal = (1.0 - ratioVal) * scale2 + ratioVal * scale3;
		}
		else
		{
			float ratioVal = (float)(a - idx3) / athird;
			scVal = (1.0 - ratioVal) * scale3 + ratioVal * scale4;
		}

		//int randVal = (rand() % 100) + 50;
		//float scaleVal = (float)randVal / 100.0;
		randomScale.push_back(scVal);
	}

	A2DVector ctr = _layer_center;
	for (int a = 0; a < _massList.size(); a++)
	{
		A2DVector pos2D = _massList[a]._pos.GetA2DVector();
		pos2D -= ctr;
		
		int lyr = _massList[a]._layer_idx;
		pos2D *= randomScale[lyr];

		pos2D += ctr;

		_massList[a]._pos.SetXY(pos2D.x, pos2D.y);

	}
}

void  AnElement::CreateHelix()
{
	float ggg = 6.28318530718 * 2.0;

	int randVal = rand() % 2;
	if (randVal == 0)
	{
		ggg = -ggg;
	}

	for (int a = 0; a < _massList.size(); a++)
	{
		//if (a % 11 == 0) { continue; }
		A2DVector pos(_massList[a]._pos._x, _massList[a]._pos._y);
		int curLayer = _massList[a]._layer_idx;
		float radAngle = (ggg / (float)SystemParams::_num_layer) * (float)curLayer;
		A2DVector rotPos = UtilityFunctions::Rotate(pos, _layer_center, radAngle);
		_massList[a]._pos._x = rotPos.x;
		_massList[a]._pos._y = rotPos.y;
	}
}

// INTERPOLATION
/*void AnElement::Interp_UpdateLayerBoundaries()
{
	// per layer boundary
	for (int a = 0; a < _interp_massList.size(); a++)
	{
		int perLayerIdx = a % _numPointPerLayer;
		if (perLayerIdx < _numBoundaryPointPerLayer)
		{
			int layerIdx = _interp_massList[a]._layer_idx;
			_interp_per_layer_boundary[layerIdx][perLayerIdx] = _interp_massList[a]._pos.GetA2DVector();
		}
	}
}*/

std::vector<std::vector<A2DVector>> AnElement::GetBilinearInterpolatedArt(std::vector<std::vector<A2DVector>> triangles)
{
	int art_sz = _arts.size();
	std::vector<std::vector<A2DVector>> transformedArts;
	ABary bary(0, 0, 0);
	for (unsigned int a = 0; a < art_sz; a++)
	{
		int art_sz_2 = _arts[a].size();
		std::vector<A2DVector> a_array;
		for (unsigned int b = 0; b < art_sz_2; b++)
		{
			bary = _baryCoords[a][b];
			A2DVector pt = triangles[_arts2Triangles[a][b]][0] * bary._u +
				           triangles[_arts2Triangles[a][b]][1] * bary._v + 
						   triangles[_arts2Triangles[a][b]][2] * bary._w;
			a_array.push_back(pt);
		}
		transformedArts.push_back(a_array);
	}
	return transformedArts;
}

void AnElement::BiliniearInterpolationTriangle(std::vector<std::vector<A3DVector>>& triangleA,      // 3D
											   std::vector<std::vector<A3DVector>>& triangleB,      // 3D
											   std::vector<std::vector<A2DVector>>& triangleInterp, // 2D
											   float interVal)
{
	for (unsigned int a = 0; a < _numTrianglePerLayer; a++)
	{
		std::vector<A2DVector> aTri2D;
		for (unsigned int b = 0; b < 3; b++)
		{
			A3DVector pt1      = triangleA[a][b];
			A3DVector pt1_next = triangleB[a][b];
			A3DVector dir1     = pt1.DirectionTo(pt1_next);
			float dir1_len;
			A3DVector dir1_unit;
			dir1.GetUnitAndDist(dir1_unit, dir1_len);
			aTri2D.push_back( (pt1 + (dir1_unit * dir1_len * interVal)).GetA2DVector() );
		}
		triangleInterp.push_back(aTri2D);
	}
}

void AnElement::BiliniearInterpolation(std::vector<A3DVector>& boundaryA, std::vector<A3DVector>& boundaryB, std::vector<A3DVector>& boundaryInterp, float interVal)
{
	for (int b = 0; b < _numBoundaryPointPerLayer; b++)
	{
		A3DVector pt1 = boundaryA[b];
		A3DVector pt1_next = boundaryB[b];
		A3DVector dir1 = pt1.DirectionTo(pt1_next);
		float dir1_len;
		A3DVector dir1_unit;
		dir1.GetUnitAndDist(dir1_unit, dir1_len);
		boundaryInterp[b] = pt1 + (dir1_unit * dir1_len * interVal);
	}
}

void AnElement::CalculateLayerTriangles_Drawing()
{
	// clear
	_per_layer_triangle_drawing.clear();
	
	// all actual triangles;
	std::vector<std::vector<A3DVector>> allActualTriangles3D;
	for (unsigned int a = 0; a < _triangles.size(); a++)
	{
		std::vector<A3DVector> tri;
		tri.push_back( _massList[_triangles[a].idx0]._pos );
		tri.push_back( _massList[_triangles[a].idx1]._pos );
		tri.push_back( _massList[_triangles[a].idx2]._pos );
		allActualTriangles3D.push_back(tri);
	}
	
	// ----- iter variables -----
	float z_iter = 0;        // negative
	int png_iter = 0;        // frames (png)
	int tube_layer_iter = 0; // layers of the tube

	// ----- stuff -----
	int max_num_png = SystemParams::_num_png_frame - 1;
	float z_step = SystemParams::_upscaleFactor / ((float)max_num_png);

	// first one
	{
		std::vector<std::vector<A2DVector>> first_layer_tri;
		for (unsigned int a = 0; a < _numTrianglePerLayer; a++)
		{
			std::vector<A2DVector> tri;
			tri.push_back(allActualTriangles3D[a][0].GetA2DVector());
			tri.push_back(allActualTriangles3D[a][1].GetA2DVector());
			tri.push_back(allActualTriangles3D[a][2].GetA2DVector());
			first_layer_tri.push_back(tri);
		}
		_per_layer_triangle_drawing.push_back(first_layer_tri);
	}
	z_iter -= z_step;
	png_iter++;

	std::vector<std::vector<A3DVector>>::const_iterator first_iter;
	std::vector<std::vector<A3DVector>>::const_iterator last_iter;
	float min_upscale_factor = -SystemParams::_upscaleFactor;
	int max_layer_iter = SystemParams::_num_layer - 1;
	//std::vector<std::vector<A2DVector>> a_layer_triangle_temp = _arts; // temporary?
	while (z_iter > min_upscale_factor && tube_layer_iter < max_layer_iter && png_iter < max_num_png) // todo: FIX ME
	{
		float cur_layer_z_pos = _per_layer_boundary[tube_layer_iter][0]._z;     // negative
		float next_layer_z_pos = _per_layer_boundary[tube_layer_iter + 1][0]._z; // negative

		if (z_iter < cur_layer_z_pos && z_iter > next_layer_z_pos)
		{
			// create a new frame!
			float l_2_l = -(next_layer_z_pos - cur_layer_z_pos); // positive, layer to layer dist
			float p_2_l = -(z_iter - cur_layer_z_pos);           // positive, png to layer dist
			float interp_ratio = p_2_l / l_2_l;

			// interpolation code here
			// dammi* the wost code I've ever had
			
			//std::vector<std::vector<A3DVector>> triangleA(&allActualTriangles3D[(tube_layer_iter) * _numTrianglePerLayer],     &allActualTriangles3D[(tube_layer_iter + 1) * _numTrianglePerLayer]); // 3D, tube_layer_iter
			first_iter = allActualTriangles3D.begin() + ((tube_layer_iter) * _numTrianglePerLayer);
			last_iter = allActualTriangles3D.begin() + ((tube_layer_iter + 1) * _numTrianglePerLayer);
			std::vector<std::vector<A3DVector>> triangleA(first_iter, last_iter);
			//std::vector<std::vector<A3DVector>> triangleB(&allActualTriangles3D[(tube_layer_iter + 1) * _numTrianglePerLayer], &allActualTriangles3D[(tube_layer_iter + 2) * _numTrianglePerLayer]); // 3D, tube_layer_iter + 1
			first_iter = allActualTriangles3D.begin() + ((tube_layer_iter + 1) * _numTrianglePerLayer);
			last_iter = allActualTriangles3D.begin() + ((tube_layer_iter + 2) * _numTrianglePerLayer);
			std::vector<std::vector<A3DVector>> triangleB(first_iter, last_iter);
			std::vector<std::vector<A2DVector>> a_layer_triangle_temp;
			BiliniearInterpolationTriangle(triangleA, triangleB, a_layer_triangle_temp, interp_ratio);
			_per_layer_triangle_drawing.push_back(a_layer_triangle_temp);

			// move on	
			z_iter -= z_step;
			png_iter++;
		}
		else if (z_iter < next_layer_z_pos)		{
			// move on			
			tube_layer_iter++;
		}
	}

	// last one
	{
		std::vector<std::vector<A2DVector>> last_layer_tri;
		int startIdx = _triangles.size() - _numTrianglePerLayer;
		for (unsigned int a = startIdx; a < _triangles.size(); a++)
		{
			std::vector<A2DVector> tri;
			tri.push_back(allActualTriangles3D[a][0].GetA2DVector());
			tri.push_back(allActualTriangles3D[a][1].GetA2DVector());
			tri.push_back(allActualTriangles3D[a][2].GetA2DVector());
			last_layer_tri.push_back(tri);
		}
		_per_layer_triangle_drawing.push_back(last_layer_tri);
	}
}

void AnElement::CalculateLayerBoundaries_Drawing()
{	
	_per_layer_boundary_drawing.clear();
	std::vector<A3DVector> a_layer_boundary_temp(_numBoundaryPointPerLayer, A3DVector()); // temporary

	// ----- iter variables -----
	float z_iter = 0;        // negative
	int png_iter = 0;      // frames (png)
	int tube_layer_iter = 0; // layers of the tube
	
	// ----- stuff -----
	int max_num_png = SystemParams::_num_png_frame - 1;
	float z_step = SystemParams::_upscaleFactor / ((float)(max_num_png - 1) );

	// ----- first one -----
	_per_layer_boundary_drawing.push_back(_per_layer_boundary[0]);
	z_iter -= z_step;
	png_iter++;
	//tube_layer_iter++; we are not sure to move onto the next one

	float min_upscale_factor = -SystemParams::_upscaleFactor;
	int max_layer_iter = SystemParams::_num_layer - 1;
	while (z_iter > min_upscale_factor && tube_layer_iter < max_layer_iter && png_iter < max_num_png) // todo: FIX ME
	{
		float cur_layer_z_pos  = _per_layer_boundary[tube_layer_iter][0]._z;     // negative
		float next_layer_z_pos = _per_layer_boundary[tube_layer_iter + 1][0]._z; // negative
		
		if (z_iter < cur_layer_z_pos && z_iter > next_layer_z_pos)
		{
			// create a new frame!
			float l_2_l = -(next_layer_z_pos - cur_layer_z_pos); // positive, layer to layer dist
			float p_2_l = -(z_iter - cur_layer_z_pos); // positive, png to layer dist
			float interp_ratio = p_2_l / l_2_l;

			BiliniearInterpolation(_per_layer_boundary[tube_layer_iter], 
				                   _per_layer_boundary[tube_layer_iter + 1], 
				                   a_layer_boundary_temp, 
				                   interp_ratio);
			_per_layer_boundary_drawing.push_back(a_layer_boundary_temp);

			// move on	
			z_iter -= z_step;
			png_iter++;

		}
		else if (z_iter < next_layer_z_pos)
		{
			// move on			
			tube_layer_iter++;

		}
		// else ???
		
	}

	// ----- last one -----
	_per_layer_boundary_drawing.push_back(_per_layer_boundary[SystemParams::_num_layer - 1]);
}

// ORIGINAL
void AnElement::UpdateLayerBoundaries()
{	
	// per layer boundary
	for (int a = 0; a < _massList.size(); a++)
	{
		int perLayerIdx = a % _numPointPerLayer;
		if (perLayerIdx < _numBoundaryPointPerLayer)
		{
			int layerIdx = _massList[a]._layer_idx;
			_per_layer_boundary[layerIdx][perLayerIdx] = _massList[a]._pos;
		}
	}
}

// back end
void AnElement::CreateStarTube(int self_idx)
{
	// for identification
	_elem_idx = self_idx;

	/*
	center = 250, 250, 0
	0, 193, 0
	172, 168, 0
	250, 12, 0
	327, 168, 0
	500, 193, 0
	375, 315, 0
	404, 487, 0
	250, 406, 0
	95, 487, 0
	125, 315, 0
	*/

	//_massList.push_back(AMass());
	float zPos = 0;
	float zOffset = -(SystemParams::_upscaleFactor / (SystemParams::_num_layer - 1));
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		int idxGap = a * 11;

		// x y z mass_idx element_idx layer_idx
		_massList.push_back(AMass(250, 250, zPos, 0 + idxGap, _elem_idx, a)); // 0 center
		_massList.push_back(AMass(0,   193, zPos, 1 + idxGap, _elem_idx, a, true)); // 1
		_massList.push_back(AMass(172, 168, zPos, 2 + idxGap, _elem_idx, a, true)); // 2
		_massList.push_back(AMass(250, 12,  zPos, 3 + idxGap, _elem_idx, a, true)); // 3
		_massList.push_back(AMass(327, 168, zPos, 4 + idxGap, _elem_idx, a, true)); // 4
		_massList.push_back(AMass(500, 193, zPos, 5 + idxGap, _elem_idx, a, true)); // 5
		_massList.push_back(AMass(375, 315, zPos, 6 + idxGap, _elem_idx, a, true)); // 6
		_massList.push_back(AMass(404, 487, zPos, 7 + idxGap, _elem_idx, a, true)); // 7
		_massList.push_back(AMass(250, 406, zPos, 8 + idxGap, _elem_idx, a, true)); // 8
		_massList.push_back(AMass(95,  487, zPos, 9 + idxGap, _elem_idx, a, true)); // 9
		_massList.push_back(AMass(125, 315, zPos, 10 + idxGap, _elem_idx, a, true)); // 10

		zPos += zOffset;
	}

	// ???
	//RandomizeLayerSize();
	//CreateHelix();

	//if (createAcrossTube)
	//{
	//	BuildAcrossTube();
	//}
	int idxOffset = 0;
	int offsetGap = 11;
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		// center to side
		/*_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 1));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 2));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 3));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 4));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 5));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 6));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 7));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 8));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 9));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 10));

																	  // pentagon
		_triEdges.push_back(AnIndexedLine(idxOffset + 10, idxOffset + 2));
		_triEdges.push_back(AnIndexedLine(idxOffset + 2, idxOffset + 4));
		_triEdges.push_back(AnIndexedLine(idxOffset + 4, idxOffset + 6));
		_triEdges.push_back(AnIndexedLine(idxOffset + 6, idxOffset + 8));
		_triEdges.push_back(AnIndexedLine(idxOffset + 8, idxOffset + 10));

																		  // side to side
		_triEdges.push_back(AnIndexedLine(idxOffset + 1, idxOffset + 2));
		_triEdges.push_back(AnIndexedLine(idxOffset + 2, idxOffset + 3));
		_triEdges.push_back(AnIndexedLine(idxOffset + 3, idxOffset + 4));
		_triEdges.push_back(AnIndexedLine(idxOffset + 4, idxOffset + 5));
		_triEdges.push_back(AnIndexedLine(idxOffset + 5, idxOffset + 6));
		_triEdges.push_back(AnIndexedLine(idxOffset + 6, idxOffset + 7));
		_triEdges.push_back(AnIndexedLine(idxOffset + 7, idxOffset + 8));
		_triEdges.push_back(AnIndexedLine(idxOffset + 8, idxOffset + 9));
		_triEdges.push_back(AnIndexedLine(idxOffset + 9, idxOffset + 10));
		_triEdges.push_back(AnIndexedLine(idxOffset + 10, idxOffset + 1));*/

		if (idxOffset > 0)
		{
			int prevOffset = idxOffset - offsetGap;

			// layer to layer
			/*_triEdges.push_back(AnIndexedLine(prevOffset, idxOffset, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 1, idxOffset + 1, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 2, idxOffset + 2, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 3, idxOffset + 3, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 4, idxOffset + 4, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 5, idxOffset + 5, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 6, idxOffset + 6, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 7, idxOffset + 7, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 8, idxOffset + 8, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 9, idxOffset + 9, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 10, idxOffset + 10, true));

			_triEdges.push_back(AnIndexedLine(prevOffset + 1, idxOffset + 2, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 2, idxOffset + 3, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 3, idxOffset + 4, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 4, idxOffset + 5, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 5, idxOffset + 6, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 6, idxOffset + 7, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 7, idxOffset + 8, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 8, idxOffset + 9, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 9, idxOffset + 10, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 10, idxOffset + 1, true));

			_triEdges.push_back(AnIndexedLine(prevOffset + 2, idxOffset + 1, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 3, idxOffset + 2, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 4, idxOffset + 3, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 5, idxOffset + 4, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 6, idxOffset + 5, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 7, idxOffset + 6, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 8, idxOffset + 7, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 9, idxOffset + 8, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 10, idxOffset + 9, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 1, idxOffset + 10, true));*/
		}

		idxOffset += offsetGap;
	}

	// 
	ResetSpringRestLengths();

	// _per_layer_points
	/*for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		_per_layer_points.push_back(std::vector<A2DVector>());
		_per_layer_boundary.push_back(std::vector<A2DVector>());
	}
	for (int a = 0; a < _massList.size(); a++)
	{
		A2DVector pt(_massList[a]._pos._x, _massList[a]._pos._y);
		int layer_idx = _massList[a]._layer_idx;
		_per_layer_points[layer_idx].push_back(pt);
	}
	// per layer boundary
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		for (int b = 1; b < 11; b++)
		{
			_per_layer_boundary[a].push_back(_per_layer_points[a][b]);
		}
	}*/
}

/*void AnElement::Interp_ResetSpringRestLengths()
{
	// interpolation, mid
	for (int a = 0; a < _interp_triEdges.size(); a++)
	{
		A2DVector p1 = _interp_massList[_interp_triEdges[a]._index0]._pos.GetA2DVector();
		A2DVector p2 = _interp_massList[_interp_triEdges[a]._index1]._pos.GetA2DVector();
		_interp_triEdges[a].SetActualOriDistance(p1.Distance(p2));
	}

	// interpolation, aux
	for (int a = 0; a < _interp_auxiliaryEdges.size(); a++)
	{
		A2DVector p1 = _interp_massList[_interp_auxiliaryEdges[a]._index0]._pos.GetA2DVector();
		A2DVector p2 = _interp_massList[_interp_auxiliaryEdges[a]._index1]._pos.GetA2DVector();
		_interp_auxiliaryEdges[a].SetActualOriDistance(p1.Distance(p2));
	}

	// interpolation, first
	// first index is from original, second index is from interpolation
	for (int a = 0; a < _timeEdgesA.size(); a++)
	{
		A2DVector p1 = _massList[_timeEdgesA[a]._index0]._pos.GetA2DVector();
		A2DVector p2 = _interp_massList[_timeEdgesA[a]._index1]._pos.GetA2DVector();
		_timeEdgesA[a].SetActualOriDistance(p1.Distance(p2));
	}

	// interpolation, last
	// first index is from interpolation, second index is from original
	for (int a = 0; a < _timeEdgesB.size(); a++)
	{
		A2DVector p1 = _interp_massList[_timeEdgesB[a]._index0]._pos.GetA2DVector();
		A2DVector p2 = _massList[_timeEdgesB[a]._index1]._pos.GetA2DVector();
		_timeEdgesB[a].SetActualOriDistance(p1.Distance(p2));
	}
}*/

void AnElement::ResetSpringRestLengths()
{
	for (int a = 0; a < _layer_springs.size(); a++)
	{	
		A3DVector p1 = _massList[_layer_springs[a]._index0]._pos;
		A3DVector p2 = _massList[_layer_springs[a]._index1]._pos;
		_layer_springs[a].SetActualOriDistance(p1.Distance(p2));
	}

	for (int a = 0; a < _time_springs.size(); a++)
	{
		A3DVector p1 = _massList[_time_springs[a]._index0]._pos;
		A3DVector p2 = _massList[_time_springs[a]._index1]._pos;
		_time_springs[a].SetActualOriDistance(p1.Distance(p2));
	}

	for (int a = 0; a < _auxiliary_springs.size(); a++)
	{
		//{
		A3DVector p1 = _massList[_auxiliary_springs[a]._index0]._pos;
		A3DVector p2 = _massList[_auxiliary_springs[a]._index1]._pos;
		_auxiliary_springs[a].SetActualOriDistance(p1.Distance(p2));
	}

	for (int a = 0; a < _neg_space_springs.size(); a++)
	{
		//{
		A3DVector p1 = _massList[_neg_space_springs[a]._index0]._pos;
		A3DVector p2 = _massList[_neg_space_springs[a]._index1]._pos;
		_neg_space_springs[a].SetActualOriDistance(p1.Distance(p2));
	}

	for (int a = 0; a < _massList.size(); a++)
	{
		_massList[a]._ori_z_pos = _massList[a]._pos._z;
	}
	
	
}

/*A2DVector  AnElement::Interp_ClosestPtOnALayer(A2DVector pt, int layer_idx)
{
	A2DVector closestPt;
	closestPt = UtilityFunctions::GetClosestPtOnClosedCurve(_interp_per_layer_boundary[layer_idx], pt);

	return closestPt;
}*/

A3DVector AnElement::ClosestPtOnATriSurface(int triIdx, A3DVector pos)
{
	_tempTri3[0] = _massList[_surfaceTriangles[triIdx].idx0]._pos;
	_tempTri3[1] = _massList[_surfaceTriangles[triIdx].idx1]._pos;
	_tempTri3[2] = _massList[_surfaceTriangles[triIdx].idx2]._pos;
	A3DVector cPt = UtilityFunctions::ClosestPointOnTriangle2(pos, _tempTri3[0], _tempTri3[1], _tempTri3[2]);
	return cPt;
}

A3DVector AnElement::ClosestPtOnTriSurfaces(std::vector<int>& triIndices, A3DVector pos)
{
	A3DVector closestPt;
	float dist = 10000000000000;
	//for (int a = 0; a < massIndices.size(); a++)
	//{
		//int idx = massIndices[a];
	for (int b = 0; b < _surfaceTriangles.size(); b++)
	{
		//std::vector<A3DVector> tri;
		_tempTri3[0] = _massList[_surfaceTriangles[b].idx0]._pos;
		_tempTri3[1] = _massList[_surfaceTriangles[b].idx1]._pos;
		_tempTri3[2] = _massList[_surfaceTriangles[b].idx2]._pos;

		//A3DVector cPt = UtilityFunctions::ClosestPointOnTriangle(_tempTri3, pos);
		A3DVector cPt = UtilityFunctions::ClosestPointOnTriangle2(pos, _tempTri3[0], _tempTri3[1], _tempTri3[2]);
		float d = cPt.DistanceSquared(pos);
		if (d < dist)
		{
			dist = d;
			closestPt = cPt;
		}
	}
	//}

	return closestPt;
}

/*A2DVector AnElement::ClosestPtOnALayer(A2DVector pt, int layer_idx)
{
	A2DVector closestPt;
	closestPt = UtilityFunctions::GetClosestPtOnClosedCurve(_per_layer_boundary[layer_idx], pt);
	return closestPt;
}*/

/*void AnElement::Interp_SolveForSprings2D()
{
	A2DVector pt0;
	A2DVector pt1;
	A2DVector dir;
	A2DVector eForce;
	float dist = 0;
	float diff = 0;
	float k = 0;

	//std::cout << _interp_triEdges[0]._dist << " " << _interp_triEdges[0]._oriDist << "\n";

	for (unsigned int a = 0; a < _interp_triEdges.size(); a++)
	{
		int idx0 = _interp_triEdges[a]._index0;
		int idx1 = _interp_triEdges[a]._index1;

		pt0 = _interp_massList[idx0]._pos.GetA2DVector();
		pt1 = _interp_massList[idx1]._pos.GetA2DVector();

		if (_interp_triEdges[a]._isLayer2Layer) { k = SystemParams::_k_time_edge; }
		else { k = SystemParams::_k_edge; } 

		dir = pt0.DirectionTo(pt1);
		dir = dir.Norm();
		dist = pt0.Distance(pt1);
		diff = dist - _interp_triEdges[a]._dist;

		eForce = dir * k *  diff;

		
		{
			_interp_massList[idx0]._edgeForce += A3DVector(eForce.x, eForce.y, 0);	// _massList[idx0]._distToBoundary;
			_interp_massList[idx1]._edgeForce -= A3DVector(eForce.x, eForce.y, 0);	// _massList[idx1]._distToBoundary;
		}
	}

	for (unsigned int a = 0; a < _interp_auxiliaryEdges.size(); a++)
	{
		int idx0 = _interp_auxiliaryEdges[a]._index0;
		int idx1 = _interp_auxiliaryEdges[a]._index1;

		pt0 = _interp_massList[idx0]._pos.GetA2DVector();
		pt1 = _interp_massList[idx1]._pos.GetA2DVector();


		k = SystemParams::_k_edge;

		dir = pt0.DirectionTo(pt1);
		dir = dir.Norm();
		dist = pt0.Distance(pt1);
		diff = dist - _interp_auxiliaryEdges[a]._dist;

		eForce = dir * k *  diff;

		
		{
			_interp_massList[idx0]._edgeForce += A3DVector(eForce.x, eForce.y, 0);	// _massList[idx0]._distToBoundary;
			_interp_massList[idx1]._edgeForce -= A3DVector(eForce.x, eForce.y, 0);	// _massList[idx1]._distToBoundary;
		}
	}
}*/

// 33333333333333333333333333333333
void AnElement::SolveForSprings3D_Linear()
{
	//A3DVector pt0;
	//A3DVector pt1;
	A3DVector dir;
	A3DVector dir_not_unit;
	A3DVector eForce;
	float dist = 0;
	float diff = 0;
	float k = 0;
	int idx0, idx1;

	// TODO: Nasty code here
	//float scale_threshold = 1.0f;
	//float magic_number = 3.0f;

	// ----- 00000 Layer Spring -----
	int tr_sz = _layer_springs.size();
	k = SystemParams::_k_edge;
	for (unsigned int a = 0; a < tr_sz; a++)
	{
		idx0 = _layer_springs[a]._index0;
		idx1 = _layer_springs[a]._index1;

		dir_not_unit = _massList[idx0]._pos.DirectionTo(_massList[idx1]._pos);
		dir_not_unit.GetUnitAndDist(dir, dist);

		diff = dist - _layer_springs[a]._dist;

		_layer_springs[a]._diff = diff; // debug delete me
		_layer_springs[a]._k_debug = k;// debug delete me
		_layer_springs[a]._mag = dist; // debug delete me
		_layer_springs[a]._dir = dir;// debug delete me
		
		_layer_springs[a]._signval_debug = 0; // debug delete me

		eForce = dir * k *  diff;

		_massList[idx0]._edgeForce += eForce;
		_massList[idx1]._edgeForce -= eForce;
	}

	// ----- 11111 Time Spring -----
	tr_sz = _time_springs.size();
	k = SystemParams::_k_time_edge;
	for (unsigned int a = 0; a < tr_sz; a++)
	{
		idx0 = _time_springs[a]._index0;
		idx1 = _time_springs[a]._index1;

		dir_not_unit = _massList[idx0]._pos.DirectionTo(_massList[idx1]._pos);
		dir_not_unit.GetUnitAndDist(dir, dist);

		diff = dist - _time_springs[a]._dist;

		_time_springs[a]._diff = diff; // debug delete me
		_time_springs[a]._k_debug = k; // debug delete me
		_time_springs[a]._mag = dist; // debug delete me
		_time_springs[a]._dir = dir;// debug delete me
		
		_time_springs[a]._signval_debug = 0; // debug delete me

		eForce = dir * k *  diff;

		_massList[idx0]._edgeForce += eForce;
		_massList[idx1]._edgeForce -= eForce;
	}

	// ----- 22222 Auxiliary Spring -----
	int aux_sz = _auxiliary_springs.size();
	k = SystemParams::_k_edge;
	for (unsigned int a = 0; a < aux_sz; a++)
	{
		idx0 = _auxiliary_springs[a]._index0;
		idx1 = _auxiliary_springs[a]._index1;

		dir_not_unit = _massList[idx0]._pos.DirectionTo(_massList[idx1]._pos);
		dir_not_unit.GetUnitAndDist(dir, dist);

		diff = dist - _auxiliary_springs[a]._dist;

		_auxiliary_springs[a]._diff = diff; // debug delete me
		_auxiliary_springs[a]._k_debug = k; // debug delete me
		_auxiliary_springs[a]._mag = dist; // debug delete me
		_auxiliary_springs[a]._dir = dir;// debug delete me
		
		_auxiliary_springs[a]._signval_debug = 0; // debug delete me

		eForce = dir * k *  diff;

		_massList[idx0]._edgeForce += eForce;
		_massList[idx1]._edgeForce -= eForce;
	}

	// ----- 33333 Negative Space Spring -----
	k = SystemParams::_k_neg_space_edge;
	int neg_sp_sz = _neg_space_springs.size();
	for (unsigned int a = 0; a < neg_sp_sz; a++)
	{
		idx0 = _neg_space_springs[a]._index0;
		idx1 = _neg_space_springs[a]._index1;

		dir_not_unit = _massList[idx0]._pos.DirectionTo(_massList[idx1]._pos);
		dir_not_unit.GetUnitAndDist(dir, dist);

		diff = dist - _neg_space_springs[a]._dist;

		_neg_space_springs[a]._diff = diff; // debug delete me
		_neg_space_springs[a]._k_debug = k; // debug delete me
		_neg_space_springs[a]._mag = dist; // debug delete me
		_neg_space_springs[a]._dir = dir;// debug delete me

		_neg_space_springs[a]._signval_debug = 0; // debug delete me

		eForce = dir * k *  diff;

		_massList[idx0]._edgeForce += eForce;
		_massList[idx1]._edgeForce -= eForce;
	}
}


// 33333333333333333333333333333333
void AnElement::SolveForSprings3D()
{
	//A3DVector pt0;
	//A3DVector pt1;
	A3DVector dir;
	A3DVector dir_not_unit;
	A3DVector eForce;
	float dist = 0;
	float diff = 0;
	float k = 0;
	int idx0, idx1;

	// TODO: Nasty code here
	//float scale_threshold = 1.0f;
	//float magic_number = 3.0f;

	// for squared forces
	float signVal = 1;

	// ----- 00000 Layer Spring -----
	int tr_sz = _layer_springs.size();
	k = SystemParams::_k_edge;
	for (unsigned int a = 0; a < tr_sz; a++)
	{
		idx0 = _layer_springs[a]._index0;
		idx1 = _layer_springs[a]._index1;
		
		dir_not_unit = _massList[idx0]._pos.DirectionTo(_massList[idx1]._pos);
		dir_not_unit.GetUnitAndDist(dir, dist);

		diff = dist - _layer_springs[a]._dist;
				
		_layer_springs[a]._diff = diff; // debug delete me
		_layer_springs[a]._k_debug = k;// debug delete me
		_layer_springs[a]._mag = dist; // debug delete me
		_layer_springs[a]._dir = dir;// debug delete me

		// squared version
		signVal = 1;
		if (diff < 0) { signVal = -1; }

		_layer_springs[a]._signval_debug = signVal; // debug delete me

		eForce = dir * k *  diff * diff * signVal;

		_massList[idx0]._edgeForce_cpu += eForce;	
		_massList[idx1]._edgeForce_cpu -= eForce;
	}

	// ----- 11111 Time Spring -----
	tr_sz = _time_springs.size();
	k = SystemParams::_k_time_edge;
	for (unsigned int a = 0; a < tr_sz; a++)
	{
		idx0 = _time_springs[a]._index0;
		idx1 = _time_springs[a]._index1;

		dir_not_unit = _massList[idx0]._pos.DirectionTo(_massList[idx1]._pos);
		dir_not_unit.GetUnitAndDist(dir, dist);

		diff = dist - _time_springs[a]._dist;
				
		_time_springs[a]._diff = diff; // debug delete me
		_time_springs[a]._k_debug = k; // debug delete me
		_time_springs[a]._mag = dist; // debug delete me
		_time_springs[a]._dir = dir;// debug delete me

		// squared version
		signVal = 1;
		if (diff < 0) { signVal = -1; }

		_time_springs[a]._signval_debug = signVal; // debug delete me

		eForce = dir * k *  diff * diff * signVal;

		_massList[idx0]._edgeForce_cpu += eForce;
		_massList[idx1]._edgeForce_cpu -= eForce;
	}

	// ----- 22222 Auxiliary Spring -----
	int aux_sz = _auxiliary_springs.size();
	k = SystemParams::_k_edge;
	for (unsigned int a = 0; a < aux_sz; a++)
	{
		idx0 = _auxiliary_springs[a]._index0;
		idx1 = _auxiliary_springs[a]._index1;		

		dir_not_unit = _massList[idx0]._pos.DirectionTo(_massList[idx1]._pos);
		dir_not_unit.GetUnitAndDist(dir, dist);

		diff = dist - _auxiliary_springs[a]._dist;
				
		_auxiliary_springs[a]._diff = diff; // debug delete me
		_auxiliary_springs[a]._k_debug = k; // debug delete me
		_auxiliary_springs[a]._mag = dist; // debug delete me
		_auxiliary_springs[a]._dir = dir;// debug delete me

		// squared version
		signVal = 1;
		if (diff < 0) { signVal = -1; }

		_auxiliary_springs[a]._signval_debug = signVal; // debug delete me

		eForce = dir * k *  diff * diff * signVal;

		_massList[idx0]._edgeForce_cpu += eForce;
		_massList[idx1]._edgeForce_cpu -= eForce;
	}

	// ----- 33333 Negative Space Spring -----
	k = SystemParams::_k_neg_space_edge;
	int neg_sp_sz = _neg_space_springs.size();
	for (unsigned int a = 0; a < neg_sp_sz; a++)
	{
		idx0 = _neg_space_springs[a]._index0;
		idx1 = _neg_space_springs[a]._index1;

		dir_not_unit = _massList[idx0]._pos.DirectionTo(_massList[idx1]._pos);
		dir_not_unit.GetUnitAndDist(dir, dist);

		diff = dist - _neg_space_springs[a]._dist;

		_neg_space_springs[a]._diff = diff; // debug delete me
		_neg_space_springs[a]._k_debug = k; // debug delete me
		_neg_space_springs[a]._mag = dist; // debug delete me
		_neg_space_springs[a]._dir = dir;// debug delete me

		// squared version
		signVal = 1;
		if (diff < 0) { signVal = -1; }

		_neg_space_springs[a]._signval_debug = signVal; // debug delete me

		eForce = dir * k *  diff * diff * signVal;

		_massList[idx0]._edgeForce_cpu += eForce;
		_massList[idx1]._edgeForce_cpu -= eForce;
	}
}


/*void AnElement::UpdatePerLayerBoundaryOgre3D()
{
	_debug_lines->clear();

	float zOffset = -(SystemParams::_upscaleFactor / (SystemParams::_num_layer - 1));

	for (int b = 0; b < SystemParams::_num_layer; b++) // iterate layer
	{
		float zPos = b * zOffset;

		int boundary_sz = _per_layer_boundary[b].size();
		for (int c = 0; c < boundary_sz; c++) // iterate point
		{
			if (c == 0)
			{
				A3DVector pt1 = _per_layer_boundary[b][boundary_sz - 1];
				A3DVector pt2 = _per_layer_boundary[b][c];
				_debug_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
				_debug_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));

				continue;
			}

			A3DVector pt1 = _per_layer_boundary[b][c - 1];
			A3DVector pt2 = _per_layer_boundary[b][c];
			_debug_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_debug_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}

	}
	//}
	_debug_lines->update();
}*/

void AnElement::UpdateClosestPtsDisplayOgre3D()
{
	_closet_pt_lines->clear();
	_closet_pt_approx_lines->clear();

	//_closet_pt_lines_back->clear();
	//_closet_pt_approx_lines_back->clear();

	//int layerStop = SystemParams::_num_layer / 2; // delete
	//if (layerStop < 0) { layerStop = 0; } // delete

	float offst = 2;

	if(SystemParams::_show_exact_repulsion_forces)
	{
		for (int b = 0; b < _massList.size(); b++)
		{
			if (SystemParams::_layer_slider_int >= 0 && _massList[b]._layer_idx != SystemParams::_layer_slider_int) { continue; }

			if (_massList[b]._c_pts_fill_size == 0) { continue; }

			A3DVector pt1 = _massList[b]._pos;
			_closet_pt_lines->addPoint(Ogre::Vector3(pt1._x - offst, pt1._y, pt1._z));
			_closet_pt_lines->addPoint(Ogre::Vector3(pt1._x + offst, pt1._y, pt1._z));
			_closet_pt_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y - offst, pt1._z));
			_closet_pt_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y + offst, pt1._z));

			for (int c = 0; c < _massList[b]._c_pts_fill_size; c++)
			{
				A3DVector pt2(_massList[b]._c_pts[c]._x, _massList[b]._c_pts[c]._y, _massList[b]._c_pts[c]._z);

				_closet_pt_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
				_closet_pt_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
			}
		}
	}

	if (SystemParams::_show_approx_repulsion_forces)
	{
		for (int b = 0; b < _massList.size(); b++)
		{
			if (SystemParams::_layer_slider_int >= 0 && _massList[b]._layer_idx != SystemParams::_layer_slider_int) { continue; }

			A3DVector pt1 = _massList[b]._pos;
			for (int c = 0; c < _massList[b]._c_pts_approx_fill_size; c++)
			{
				A3DVector pt2(_massList[b]._c_pts_approx[c].first._x, _massList[b]._c_pts_approx[c].first._y, _massList[b]._c_pts_approx[c].first._z);

				_closet_pt_approx_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
				_closet_pt_approx_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
			}
		}
	}

	_closet_pt_lines->update();
	_closet_pt_approx_lines->update();

	//_closet_pt_lines_back->update();
	//_closet_pt_approx_lines_back->update();
}

int AnElement::GetUnsharedVertexIndex(AnIdxTriangle tri, AnIndexedLine edge)
{
	if (tri.idx0 != edge._index0 && tri.idx0 != edge._index1) { return tri.idx0; }

	if (tri.idx1 != edge._index0 && tri.idx1 != edge._index1) { return tri.idx1; }

	if (tri.idx2 != edge._index0 && tri.idx2 != edge._index1) { return tri.idx2; }

	return -1;
}

std::vector<AnIndexedLine> AnElement::CreateBendingSprings(std::vector<AMass>& mList,
														   const std::vector<AnIdxTriangle>& tris,
	                                                       const std::vector<AnIndexedLine>& tEdges,
														   const std::vector<std::vector<int>>& e2t)
{
	std::vector<AnIndexedLine> auxiliaryEdges;
	for (unsigned a = 0; a < e2t.size(); a++)
	{
		if (e2t[a].size() != 2) { continue; }

		int idx1 = GetUnsharedVertexIndex(tris[e2t[a][0]], tEdges[a]);
		if (idx1 < 0) { continue; }

		int idx2 = GetUnsharedVertexIndex(tris[e2t[a][1]], tEdges[a]);
		if (idx2 < 0) { continue; }

		AnIndexedLine anEdge(idx1, idx2);
		A2DVector pt1 = mList[idx1]._pos.GetA2DVector();
		A2DVector pt2 = mList[idx2]._pos.GetA2DVector();
		//anEdge._dist = pt1.Distance(pt2);
		float d = pt1.Distance(pt2);
		anEdge.SetDist(d);

		// push to edge list
		auxiliaryEdges.push_back(anEdge);
	}
	return auxiliaryEdges;
}
void  AnElement::ForceAddSpring(AnIndexedLine anEdge, int triIndex, std::vector<AnIndexedLine>& tEdges, std::vector<std::vector<int>>& e2t)
{
	A3DVector pt1 = _massList[anEdge._index0]._pos;
	A3DVector pt2 = _massList[anEdge._index1]._pos;
	float d = pt1.Distance(pt2);
	anEdge.SetDist(d);

	// push to edge list
	tEdges.push_back(anEdge);

	// push to edge-to-triangle list
	std::vector<int> indices;
	indices.push_back(triIndex);
	e2t.push_back(indices);
}

void AnElement::AddSpring(AnIndexedLine anEdge, std::vector<AnIndexedLine>& tSpring)
{
	A3DVector pt1 = _massList[anEdge._index0]._pos;
	A3DVector pt2 = _massList[anEdge._index1]._pos;
	float d = pt1.Distance(pt2);
	anEdge.SetDist(d);

	// push to edge list
	tSpring.push_back(anEdge);
}

bool AnElement::TryToAddSpring(AnIndexedLine anEdge, int triIndex, std::vector<AnIndexedLine>& tEdges, std::vector<std::vector<int>>& e2t)
{
	int edgeIndex = FindSpring(anEdge, tEdges);
	if (edgeIndex < 0)
	{
		A3DVector pt1 = _massList[anEdge._index0]._pos;
		A3DVector pt2 = _massList[anEdge._index1]._pos;
		float d = pt1.Distance(pt2);
		anEdge.SetDist(d);

		// push to edge list
		tEdges.push_back(anEdge);

		// push to edge-to-triangle list
		std::vector<int> indices;
		indices.push_back(triIndex);
		e2t.push_back(indices);

		return true;
	}

	//// push to edge-to-triangle list
	e2t[edgeIndex].push_back(triIndex);

	return false;
}


// triangle edges
int AnElement::FindSpring(AnIndexedLine anEdge, std::vector<AnIndexedLine>& tEdges)
{
	for (unsigned int a = 0; a < tEdges.size(); a++)
	{
		if (tEdges[a]._index0 == anEdge._index0 &&
			tEdges[a]._index1 == anEdge._index1)
		{
			return a;
		}

		if (tEdges[a]._index1 == anEdge._index0 &&
			tEdges[a]._index0 == anEdge._index1)
		{
			return a;
		}
	}

	return -1;
}

/*bool AnElement::Interp_HasOverlap()
{
	for (int a = 0; a < _interp_massList.size(); a++)
	{
		if (_interp_massList[a]._is_inside)
			return true;
	}
	return false;
}*/


/*void  AnElement::UpdateInterpMasses()
{
	// z axis offset
	float zOffset = -((float)SystemParams::_upscaleFactor) / ((float)SystemParams::_num_layer - 1);
	float interp_zOffset = zOffset / ((float)SystemParams::_interpolation_factor);

	float zStart = StuffWorker::_interp_iter * zOffset;

	// only 1 and two
	int numInterpolation = SystemParams::_interpolation_factor;
	for (int i = 1; i < numInterpolation; i++)
	{
		float interVal = ((float)i) / ((float)numInterpolation);


		//for (int l = 0; l < 1; l++)
		//{
		int l = StuffWorker::_interp_iter; // SET THIS !!!
		int layerOffset = l * _numPointPerLayer;
		for (int b = 0; b < _numPointPerLayer; b++)
		{
			int massIdx1 = b + layerOffset;
			int massIdx1_next = massIdx1 + _numPointPerLayer;
			A2DVector pt1 = _massList[massIdx1]._pos.GetA2DVector();
			A2DVector pt1_next = _massList[massIdx1_next]._pos.GetA2DVector();
			A2DVector dir1 = pt1.DirectionTo(pt1_next);
			float d1 = dir1.Length() * interVal;
			dir1 = dir1.Norm();
			A2DVector pt1_mid = pt1 + (dir1 * d1);

			// idx of interp pt
			int interp_idx = (i - 1) * _numPointPerLayer + b;
			_interp_massList[interp_idx]._pos._x = pt1_mid.x;
			_interp_massList[interp_idx]._pos._y = pt1_mid.y;
			_interp_massList[interp_idx]._pos._z = zStart + (interp_zOffset * i);

		} // for (int b = 0; b < _element_list[a]._numBoundaryPointPerLayer; b++)
		//} // for (int l = 0; l < SystemParams::_num_layer - 1; l++)
	} // for (int i = 1; i < numInterpolation; i++)


	
	// update _timeEdgesA
	int offVal = StuffWorker::_interp_iter * _numPointPerLayer;
	int idx = 0;
	for (int a = 0; a < _numBoundaryPointPerLayer; a++)
	{
		int idxA = offVal + a - 1; // prev
		int idxB = offVal + a + 1; // next
		if (a == 0)
		{
			idxA = offVal + _numBoundaryPointPerLayer - 1;
		}
		else if (a == _numBoundaryPointPerLayer - 1)
		{
			idxB = 0;
		}

		// first index is from original, second index is from interpolation
		_timeEdgesA[idx++]._index0 = idxA;
		_timeEdgesA[idx++]._index0 = idxB;
	}


	// update __timeEdgesB
	offVal += _numPointPerLayer;
	idx = 0;
	for (int a = 0; a < _numBoundaryPointPerLayer; a++)
	{
		int idxA = offVal + a - 1; // prev
		int idxB = offVal + a + 1; // next
		if (a == 0)
		{
			idxA = offVal + _numBoundaryPointPerLayer - 1;
		}
		else if (a == _numBoundaryPointPerLayer - 1)
		{
			idxB = 0;
		}

		// first index is from original, second index is from interpolation
		_timeEdgesB[idx++]._index1 = idxA;
		_timeEdgesB[idx++]._index1 = idxB;
	}
	
}*/


//void AnElement::EnableInterpolationMode()
//{
//	 AMass
//	/*
//	_temp_pos;
//	_temp_velocity;
//	_interpolation_mode;
//	_inter_dir;
//	_inter_dir_length;
//	*/
//
//	int numInterpolation = SystemParams::_interpolation_factor;
//	for (int i = 1; i < numInterpolation; i++) 
//	{
//		float interVal = ((float)i) / ((float)numInterpolation);
//
//		 one less layer
//		for (int l = 0; l < SystemParams::_num_layer - 1; l++)
//		{
//			int layerOffset = l * _numPointPerLayer;
//			for (int b = 0; b < _numBoundaryPointPerLayer; b++) 
//			{
//				int massIdx1 = b + layerOffset;
//				int massIdx1_next = massIdx1 + _numPointPerLayer; 
//				A2DVector pt1 = _massList[massIdx1]._pos.GetA2DVector();
//				A2DVector pt1_next = _massList[massIdx1_next]._pos.GetA2DVector();
//				A2DVector dir1 = pt1.DirectionTo(pt1_next);
//				float d1 = dir1.Length() * interVal;
//				dir1 = dir1.Norm();
//				A2DVector pt1_mid = pt1 + (dir1 * d1);
//			} // for (int b = 0; b < _element_list[a]._numBoundaryPointPerLayer; b++)
//		} // for (int l = 0; l < SystemParams::_num_layer - 1; l++)
//	} // for (int i = 1; i < numInterpolation; i++)
//
//}
//
//void AnElement::DisableInterpolationMode()
//{
//}
