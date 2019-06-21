
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
		_insideFlags.push_back(false); // TODO currently not used
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

	for (int a = 0; a < _interp_massList.size(); a++)
	{
		A2DVector pos = _interp_massList[a]._pos.GetA2DVector();
		A2DVector rotPos = UtilityFunctions::Rotate(pos, _layer_center, radAngle);
		_interp_massList[a]._pos._x = rotPos.x;
		_interp_massList[a]._pos._y = rotPos.y;
	}

	ResetSpringRestLengths();
}

void AnElement::ScaleXY(float scVal)
{
	for (int a = 0; a < _massList.size(); a++)
	{
		A3DVector pos = _massList[a]._pos;
		_massList[a]._pos = A3DVector(pos._x * scVal, pos._y * scVal, pos._z);
	}

	for (int a = 0; a < _interp_massList.size(); a++)
	{
		A3DVector pos = _interp_massList[a]._pos;
		_interp_massList[a]._pos = A3DVector(pos._x * scVal, pos._y * scVal, pos._z);
	}

	ResetSpringRestLengths();
}

void AnElement::TranslateXY(float x, float y)
{

	for (int a = 0; a < _massList.size(); a++)
	{
		A3DVector pos = _massList[a]._pos;
		_massList[a]._pos = A3DVector(pos._x + x, pos._y + y, pos._z);
	}

	for (int a = 0; a < _interp_massList.size(); a++)
	{
		A3DVector pos = _interp_massList[a]._pos;
		_interp_massList[a]._pos = A3DVector(pos._x + x, pos._y + y, pos._z);
	}

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

bool AnElement::IsInside(int layer_idx, A3DVector pos, std::vector<A2DVector>& boundary_slice)
{
	int next_layer_idx = layer_idx + 1; // guaranteed exists

	float z_1 = _per_layer_boundary[layer_idx][0]._z;
	float z_2 = _per_layer_boundary[next_layer_idx][0]._z;
	float z_length = z_2 - z_1;
	float z_length_a = pos._z - z_1;
	float interVal = z_length_a / z_length;

	for (int a = 0; a < _numBoundaryPointPerLayer; a++)
	{
		A3DVector pt1 = _per_layer_boundary[layer_idx][a];
		A3DVector pt1_next = _per_layer_boundary[next_layer_idx][a];
		A3DVector dir1 = pt1.DirectionTo(pt1_next);
		A3DVector dir1_unit;
		float dir1_len;
		dir1.GetUnitAndDist(dir1_unit, dir1_len);
		_a_layer_boundary[a] = (pt1 + (dir1_unit * dir1_len * interVal)).GetA2DVector();
	}

	boundary_slice = _a_layer_boundary;
	return UtilityFunctions::InsidePolygon(_a_layer_boundary, pos._x, pos._y);
}


void AnElement::Triangularization(std::vector<A2DVector> element_path, int self_idx)
{
	// ----- element index -----
	this->_elem_idx = self_idx;
	
	// -----  why do we need bounding box? ----- 
	A2DRectangle bb = UtilityFunctions::GetBoundingBox(element_path);
	float img_length = bb.witdh;
	if (bb.height > bb.witdh) { img_length = bb.height; }
	A2DVector centerPt = bb.GetCenter();
	_layer_center = centerPt; // assign !!!

	// -----  moving to new center ----- 
	img_length += 5.0f; // triangulation error without this ?
	A2DVector newCenter = A2DVector((img_length / 2.0f), (img_length / 2.0f));
	element_path = UtilityFunctions::MovePoly(element_path, centerPt, newCenter);

	// -----  random points ----- 
	std::vector<A2DVector> randomPoints;
	CreateRandomPoints(element_path, img_length, randomPoints, this->_numBoundaryPointPerLayer);
	this->_numPointPerLayer = randomPoints.size(); // ASSIGN

	
	// -----  triangulation ----- 
	OpenCVWrapper cvWrapper;
	std::vector<AnIdxTriangle> tempTriangles;
	cvWrapper.Triangulate(tempTriangles, randomPoints, element_path, img_length);
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
	// -----  triangulation ----- 

	// ----- interpolation triangles -----	
	for (int a = 0; a < SystemParams::_interpolation_factor - 1; a++)  // one less layer
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
	}
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
	int interp_massCounter = 0; // self_idx
	float interp_zOffset = zOffset / ((float)SystemParams::_interpolation_factor );
	for (int a = 0; a < SystemParams::_interpolation_factor - 1; a++) // one less layer
	{
		float zPos = interp_zOffset * (a + 1);
		for (int b = 0; b < randomPoints.size(); b++)
		{
			AMass m(randomPoints[b].x,     // x
					randomPoints[b].y,     // y
					zPos, // z, will be changed
					interp_massCounter++,  // self_idx
					_elem_idx,             // parent_idx
					a);                    // layer_idx
			if (b < _numBoundaryPointPerLayer) { m._is_boundary = true; }
			_interp_massList.push_back(m);
		}
	}
	// -----  generate interpolation mass ----- 

	// -----  triangle edge springs ----- 
	for (unsigned int a = 0; a < _triangles.size(); a++)
	{
		int idx0 = _triangles[a].idx0;
		int idx1 = _triangles[a].idx1;
		int idx2 = _triangles[a].idx2;

		TryToAddTriangleEdge(AnIndexedLine(idx0, idx1), a, _triEdges, _edgeToTri); // 0 - 1		
		TryToAddTriangleEdge(AnIndexedLine(idx1, idx2), a, _triEdges, _edgeToTri); // 1 - 2		
		TryToAddTriangleEdge(AnIndexedLine(idx2, idx0), a, _triEdges, _edgeToTri); // 2 - 0

		// ----- add triangles to mass -----
		AnIdxTriangle tri(idx0, idx1, idx2);
		_massList[idx0]._triangles.push_back(tri);
		_massList[idx1]._triangles.push_back(tri);
		_massList[idx2]._triangles.push_back(tri);
	}

	// ----- interpolation triangle edge springs ----- 
	for (unsigned int a = 0; a < _interp_triangles.size(); a++)
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
	}

	// ----- bending springs ----- 
	_auxiliaryEdges = CreateBendingSprings(_massList, _triangles, _triEdges, _edgeToTri);
	// ----- bending springs ----- 

	// ----- interpolation bending springs ----- 
	_interp_auxiliaryEdges = CreateBendingSprings(_interp_massList, _interp_triangles, _interp_triEdges, _interp_edgeToTri);
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
	for (int a = 0; a < SystemParams::_num_layer - 1; a++)
	{
		int massIdxOffset1 = a * randomPoints.size();
		int massIdxOffset2 = massIdxOffset1 + randomPoints.size();
		for (int b = 0; b < _numBoundaryPointPerLayer; b++)
		{
			int cur_1 = b + massIdxOffset1; // layer t
			int cur_2 = b + massIdxOffset2; // layer t + 1

			int next_1 = b + 1 + massIdxOffset1; // layer t
			int next_2 = b + 1 + massIdxOffset2; // layer t + 1

			// layer_idx --> you want to know which layer a triangle belongs to
			int layer_idx = min_of(_massList[cur_1]._layer_idx, 
				                   _massList[cur_2]._layer_idx, 
				                   _massList[next_1]._layer_idx, 
				                   _massList[next_2]._layer_idx);

			// cur_1 next_1 cur_2
			AnIdxTriangle tri1(cur_1, next_1, cur_2, layer_idx);
			_timeTriangles.push_back(tri1);
			_massList[cur_1]._timeTriangles.push_back(tri1);
			_massList[next_1]._timeTriangles.push_back(tri1);
			_massList[cur_2]._timeTriangles.push_back(tri1);

			// next_1 next_2 cur_2
			AnIdxTriangle tri2(next_1, next_2, cur_2, layer_idx);
			_timeTriangles.push_back(tri2);
			_massList[next_1]._timeTriangles.push_back(tri2);
			_massList[next_2]._timeTriangles.push_back(tri2);
			_massList[cur_2]._timeTriangles.push_back(tri2);
		}
	}
	std::cout << "_timeTriangles size = " << _timeTriangles.size() << "\n";
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
	// cross pattern
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

			if(!TryToAddTriangleEdge(AnIndexedLine(b + massIdxOffset1, idxA + massIdxOffset2, true), -1, _triEdges, _edgeToTri))
			{
				std::cout << "_triEdges error 1\n";
			}

			if(!TryToAddTriangleEdge(AnIndexedLine(b + massIdxOffset1, idxB + massIdxOffset2, true), -1, _triEdges, _edgeToTri))
			{
				std::cout << "_triEdges error 2\n";
			}

		}
	}
	// -----  triangle edge springs ----- 

	// ----- interpolation cross pattern -----
	// ----- FIRST -----
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

		// first index is from original, second index is from interpolation
		ForceAddTriangleEdge(AnIndexedLine(a, idxA, true), -1, _timeEdgesA, _interp_edgeToTriA);
		ForceAddTriangleEdge(AnIndexedLine(a, idxB, true), -1, _timeEdgesA, _interp_edgeToTriA);
	}

	// ----- MID -----
	for (int a = 0; a < SystemParams::_interpolation_factor - 2; a++) // two less
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
	}
	// ----- END -----
	int interp_factor = (SystemParams::_interpolation_factor - 2) * _numPointPerLayer;
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
	}

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
	}

	// ----- some precomputation or interpolation ----- 
	for (int a = 0; a < SystemParams::_interpolation_factor - 1; a++)
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
	}
}

void AnElement::CalculateRestStructure()
{
	UpdateLayerBoundaries(); // update per_layer_boundary

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
	for (unsigned int a = 0; a < _triEdges.size(); a++)
	{
		A2DVector p1 = _rest_mass_pos_array[_triEdges[a]._index0].GetA2DVector();
		A2DVector p2 = _rest_mass_pos_array[_triEdges[a]._index1].GetA2DVector();
		_triEdges[a].SetActualOriDistance(p1.Distance(p2));
	}

	for (unsigned int a = 0; a < _auxiliaryEdges.size(); a++)
	{
		A2DVector p1 = _rest_mass_pos_array[_auxiliaryEdges[a]._index0].GetA2DVector();
		A2DVector p2 = _rest_mass_pos_array[_auxiliaryEdges[a]._index1].GetA2DVector();
		_auxiliaryEdges[a].SetActualOriDistance(p1.Distance(p2));
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

	_color = MyColor(rVal * 255, gVal * 255, bVal * 255);

	/*_material = Ogre::MaterialManager::getSingleton().getByName(materialName)->clone(materialName + std::to_string(_self_idx));
	_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(rVal, gVal, bVal, 1));

	// clone material
	//_material = Ogre::MaterialManager::getSingleton().getByName(materialName)->clone(materialName + std::to_string(_self_idx));
	_tubeObject = _sceneMgr->createManualObject(name);
	_tubeObject->setDynamic(true);

	UpdateMeshOgre3D();

	if (_sceneNode) _sceneNode->attachObject(_tubeObject);
	else std::cout << "_sceneNode is null\n";*/

	// material
	Ogre::MaterialPtr line_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("ElementLines" + std::to_string(_elem_idx));	
	line_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(rVal, gVal, bVal, 1));
	
	// ---------- springs ----------
	/*_spring_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
	for (int a = 0; a < _triEdges.size(); a++)
	{
		AnIndexedLine ln = _triEdges[a];
		//if (ln._isLayer2Layer) { continue; }

		A3DVector pt1 = _massList[ln._index0]._pos;
		A3DVector pt2 = _massList[ln._index1]._pos;
		_spring_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
		_spring_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
	}
	for (int a = 0; a < _auxiliaryEdges.size(); a++)
	{
		AnIndexedLine ln = _auxiliaryEdges[a];
		
		A3DVector pt1 = _massList[ln._index0]._pos;
		A3DVector pt2 = _massList[ln._index1]._pos;
		_spring_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
		_spring_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
	}
	_spring_lines->update();
	_springNode = _sceneMgr->getRootSceneNode()->createChildSceneNode("SpringNode" + std::to_string(_elem_idx));
	_springNode->attachObject(_spring_lines);*/
	// ---------- springs ----------

	// ---------- boundary ----------
	//In the initialization somewhere, create the initial lines object :
	_debug_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
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
			_debug_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_debug_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
	}	
	_debug_lines->update();
	_debugNode = _sceneMgr->getRootSceneNode()->createChildSceneNode("debug_lines_" + std::to_string(_elem_idx));
	_debugNode->attachObject(_debug_lines);
	// ---------- boundary ----------





	// ---------- debug	----------
	if (_dock_mass_idx.size() > 0)
	{
		Ogre::MaterialPtr line_material_2 = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("DockLines" + std::to_string(_elem_idx));
		line_material_2->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));
		_debug_lines_2 = new DynamicLines(line_material_2, Ogre::RenderOperation::OT_LINE_LIST);

		for (int a = 0; a < _dock_mass_idx.size(); a++)
		{
			int massIdx = _dock_mass_idx[a];
			_debug_lines_2->addPoint(Ogre::Vector3(_massList[massIdx]._pos._x, _massList[massIdx]._pos._y, _massList[massIdx]._pos._z));
			_debug_lines_2->addPoint(Ogre::Vector3(_massList[massIdx]._dockPoint._x, _massList[massIdx]._dockPoint._y, _massList[massIdx]._dockPoint._z));
		}

		_debug_lines_2->update();
		_debugNode_2 = _sceneMgr->getRootSceneNode()->createChildSceneNode("debug_lines_2_" + std::to_string(_elem_idx));
		_debugNode_2->attachObject(_debug_lines_2);
	}

	// ---------- debug	----------
	/////Ogre::MaterialPtr line_material_3 = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("InterpLines" + std::to_string(_elem_idx));
	/////line_material_3->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(0, 0, 1, 1));
	/////_debug_lines_3 = new DynamicLines(line_material_3, Ogre::RenderOperation::OT_LINE_LIST);
	/*_time_springs_debug_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
	// boundary only
	for (int l = 0; l < SystemParams::_interpolation_factor - 1; l++)
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
			A3DVector pt1 = _interp_massList[massIdx1]._pos;
			A3DVector pt2 = _interp_massList[massIdx2]._pos;
			_time_springs_debug_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_time_springs_debug_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
	}
	_time_springs_debug_lines->update();
	_time_springs_debug_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("_time_springs_debug_node_" + std::to_string(_elem_idx));
	_time_springs_debug_node->attachObject(_time_springs_debug_lines);*/

	// mid
	/*for (int a = 0; a < _interp_triEdges.size(); a++)
	{
		if (_interp_triEdges[a]._isLayer2Layer) { continue; }

		int idx1 = _interp_triEdges[a]._index0;
		int idx2 = _interp_triEdges[a]._index1;
		_debug_lines_3->addPoint(Ogre::Vector3(_interp_massList[idx1]._pos._x, _interp_massList[idx1]._pos._y, _interp_massList[idx1]._pos._z));
		_debug_lines_3->addPoint(Ogre::Vector3(_interp_massList[idx2]._pos._x, _interp_massList[idx2]._pos._y, _interp_massList[idx2]._pos._z));
	}*/
	/*for (int a = 0; a < _interp_auxiliaryEdges.size(); a++)
	{
		int idx1 = _interp_auxiliaryEdges[a]._index0;
		int idx2 = _interp_auxiliaryEdges[a]._index1;
		_debug_lines_3->addPoint(Ogre::Vector3(_interp_massList[idx1]._pos._x, _interp_massList[idx1]._pos._y, _interp_massList[idx1]._pos._z));
		_debug_lines_3->addPoint(Ogre::Vector3(_interp_massList[idx2]._pos._x, _interp_massList[idx2]._pos._y, _interp_massList[idx2]._pos._z));
	}*/
	// first 
	// first index is from original, second index is from interpolation
	/*for (int a = 0; a < _timeEdgesA.size(); a++)
	{
		int idx1 = _timeEdgesA[a]._index0;
		int idx2 = _timeEdgesA[a]._index1;
		_debug_lines_3->addPoint(Ogre::Vector3(_massList[idx1]._pos._x, _massList[idx1]._pos._y, _massList[idx1]._pos._z));
		_debug_lines_3->addPoint(Ogre::Vector3(_interp_massList[idx2]._pos._x, _interp_massList[idx2]._pos._y, _interp_massList[idx2]._pos._z));
	}
	// last
	// first index is from interpolation, second index is from original
	for (int a = 0; a < _timeEdgesB.size(); a++)
	{
		int idx1 = _timeEdgesB[a]._index0;
		int idx2 = _timeEdgesB[a]._index1;
		_debug_lines_3->addPoint(Ogre::Vector3(_interp_massList[idx1]._pos._x, _interp_massList[idx1]._pos._y, _interp_massList[idx1]._pos._z));
		_debug_lines_3->addPoint(Ogre::Vector3(_massList[idx2]._pos._x, _massList[idx2]._pos._y, _massList[idx2]._pos._z));
	}*/
	/////_debug_lines_3->update();
	/////_debugNode_3 = _sceneMgr->getRootSceneNode()->createChildSceneNode("debug_lines_3_" + std::to_string(_elem_idx));
	/////_debugNode_3->attachObject(_debug_lines_3);

	// another debug
	//Ogre::MaterialPtr line_material_4 = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("ClosestLines" + std::to_string(_elem_idx));
	//line_material_4->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));
	_time_springs_debug_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);

	for (int b = 0; b < _timeTriangles.size(); b++)
	{
		std::vector<A3DVector> tri;
		_tempTri3[0] = _massList[_timeTriangles[b].idx0]._pos;
		_tempTri3[1] = _massList[_timeTriangles[b].idx1]._pos;
		_tempTri3[2] = _massList[_timeTriangles[b].idx2]._pos;

		_time_springs_debug_lines->addPoint( Ogre::Vector3(_tempTri3[0]._x, _tempTri3[0]._y, _tempTri3[0]._z) );
		_time_springs_debug_lines->addPoint(Ogre::Vector3(_tempTri3[1]._x, _tempTri3[1]._y, _tempTri3[1]._z));

		_time_springs_debug_lines->addPoint(Ogre::Vector3(_tempTri3[1]._x, _tempTri3[1]._y, _tempTri3[1]._z));
		_time_springs_debug_lines->addPoint(Ogre::Vector3(_tempTri3[2]._x, _tempTri3[2]._y, _tempTri3[2]._z));

		_time_springs_debug_lines->addPoint(Ogre::Vector3(_tempTri3[2]._x, _tempTri3[2]._y, _tempTri3[2]._z));
		_time_springs_debug_lines->addPoint(Ogre::Vector3(_tempTri3[0]._x, _tempTri3[0]._y, _tempTri3[0]._z));
		
	}
	_time_springs_debug_lines->update();
	_time_springs_debug_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("_time_springs_debug_lines_" + std::to_string(_elem_idx));
	_time_springs_debug_node->attachObject(_time_springs_debug_lines);
	_time_springs_debug_lines->setVisible(SystemParams::_show_time_springs);



	// debug closest slice
	/*DynamicLines*    _closest_slice_lines;
	Ogre::SceneNode* _closest_slice_node;*/
	Ogre::MaterialPtr line_material_asdf = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("line_material_asdf" + std::to_string(_elem_idx));
	line_material_asdf->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(1, 0, 0, 1));

	_closest_slice_lines = new DynamicLines(line_material_asdf, Ogre::RenderOperation::OT_LINE_LIST);

	_closest_slice_lines->update();
	_closest_slice_node = _sceneMgr->getRootSceneNode()->createChildSceneNode("_closest_slice_lines_" + std::to_string(_elem_idx));
	_closest_slice_node->attachObject(_closest_slice_lines);
}

void AnElement::UpdateClosestSliceOgre3D()
{
	// debug closest slice
	/*DynamicLines*    _closest_slice_lines;
	Ogre::SceneNode* _closest_slice_node;*/

	_closest_slice_lines->clear();

	for (int a = 0; a < _massList.size(); a++)
	{
		if (_massList[a]._is_boundary && _massList[a]._is_inside)
		{
			std::vector<A2DVector> slice_array = _massList[a]._closest_boundary_slice;
			float z_pos = _massList[a]._pos._z;

			for (int i = 0; i < slice_array.size(); i++)
			{
				int next_i = i + 1;
				if (i == slice_array.size() - 1)
				{
					next_i = 0;
				}
				_closest_slice_lines->addPoint(Ogre::Vector3(slice_array[i].x, slice_array[i].y, z_pos));
				_closest_slice_lines->addPoint(Ogre::Vector3(slice_array[next_i].x, slice_array[next_i].y, z_pos));
			}

		}
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
		_debug_lines_2->setPoint(idx1, Ogre::Vector3(_massList[massIdx]._pos._x, _massList[massIdx]._pos._y, _massList[massIdx]._pos._z));
		_debug_lines_2->setPoint(idx2, Ogre::Vector3(_massList[massIdx]._dockPoint._x, _massList[massIdx]._dockPoint._y, _massList[massIdx]._dockPoint._z));
	}

	_debug_lines_2->update();
}

void AnElement::UpdateBoundaryDisplayOgre3D()
{
	int idx = 0;

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
			_debug_lines->setPoint(idx++, Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_debug_lines->setPoint(idx++, Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
	}

	_debug_lines->update();

	/*idx = 0;
	for (int a = 0; a < _dock_mass_idx.size(); a++)
	{
		int massIdx = _dock_mass_idx[a];
		_debug_lines_2->setPoint(idx++, Ogre::Vector3(_massList[massIdx]._pos._x, _massList[massIdx]._pos._y, _massList[massIdx]._pos._z));
		_debug_lines_2->setPoint(idx++, Ogre::Vector3(_massList[massIdx]._dockPoint._x, _massList[massIdx]._dockPoint._y, _massList[massIdx]._dockPoint._z));
	}

	_debug_lines_2->update();*/
}

void AnElement::ShowTimeSprings(bool yesno)
{
	_time_springs_debug_node->setVisible(yesno);
}

void AnElement::UpdateDebug34Ogre3D()
{
	/*int idx = 0;*/

	/*for (int a = 0; a < _interp_triEdges.size(); a++)
	{
		if (_interp_triEdges[a]._isLayer2Layer) { continue; }

		int idx1 = _interp_triEdges[a]._index0;
		int idx2 = _interp_triEdges[a]._index1;
		_debug_lines_3->setPoint(idx++, Ogre::Vector3(_interp_massList[idx1]._pos._x, _interp_massList[idx1]._pos._y, _interp_massList[idx1]._pos._z));
		_debug_lines_3->setPoint(idx++, Ogre::Vector3(_interp_massList[idx2]._pos._x, _interp_massList[idx2]._pos._y, _interp_massList[idx2]._pos._z));
	}*/

	
	if(SystemParams::_show_time_springs)
	{
		int idx = 0;
		for (int b = 0; b < _timeTriangles.size(); b++)
		{
			std::vector<A3DVector> tri;
			_tempTri3[0] = _massList[_timeTriangles[b].idx0]._pos;
			_tempTri3[1] = _massList[_timeTriangles[b].idx1]._pos;
			_tempTri3[2] = _massList[_timeTriangles[b].idx2]._pos;

			_time_springs_debug_lines->setPoint(idx++, Ogre::Vector3(_tempTri3[0]._x, _tempTri3[0]._y, _tempTri3[0]._z));
			_time_springs_debug_lines->setPoint(idx++, Ogre::Vector3(_tempTri3[1]._x, _tempTri3[1]._y, _tempTri3[1]._z));

			_time_springs_debug_lines->setPoint(idx++, Ogre::Vector3(_tempTri3[1]._x, _tempTri3[1]._y, _tempTri3[1]._z));
			_time_springs_debug_lines->setPoint(idx++, Ogre::Vector3(_tempTri3[2]._x, _tempTri3[2]._y, _tempTri3[2]._z));

			_time_springs_debug_lines->setPoint(idx++, Ogre::Vector3(_tempTri3[2]._x, _tempTri3[2]._y, _tempTri3[2]._z));
			_time_springs_debug_lines->setPoint(idx++, Ogre::Vector3(_tempTri3[0]._x, _tempTri3[0]._y, _tempTri3[0]._z));
		}
		_time_springs_debug_lines->update();
	}

	/*for (int a = 0; a < _interp_auxiliaryEdges.size(); a++)
	{
		int idx1 = _interp_auxiliaryEdges[a]._index0;
		int idx2 = _interp_auxiliaryEdges[a]._index1;
		_debug_lines_3->setPoint(idx++, Ogre::Vector3(_interp_massList[idx1]._pos._x, _interp_massList[idx1]._pos._y, _interp_massList[idx1]._pos._z));
		_debug_lines_3->setPoint(idx++, Ogre::Vector3(_interp_massList[idx2]._pos._x, _interp_massList[idx2]._pos._y, _interp_massList[idx2]._pos._z));
	}*/
	// first 
	// first index is from original, second index is from interpolation
	/*for (int a = 0; a < _timeEdgesA.size(); a++)
	{
		int idx1 = _timeEdgesA[a]._index0;
		int idx2 = _timeEdgesA[a]._index1;
		_debug_lines_3->setPoint(idx++, Ogre::Vector3(_massList[idx1]._pos._x, _massList[idx1]._pos._y, _massList[idx1]._pos._z));
		_debug_lines_3->setPoint(idx++, Ogre::Vector3(_interp_massList[idx2]._pos._x, _interp_massList[idx2]._pos._y, _interp_massList[idx2]._pos._z));
	}
	// last
	// first index is from interpolation, second index is from original
	for (int a = 0; a < _timeEdgesB.size(); a++)
	{
		int idx1 = _timeEdgesB[a]._index0;
		int idx2 = _timeEdgesB[a]._index1;
		_debug_lines_3->setPoint(idx++, Ogre::Vector3(_interp_massList[idx1]._pos._x, _interp_massList[idx1]._pos._y, _interp_massList[idx1]._pos._z));
		_debug_lines_3->setPoint(idx++, Ogre::Vector3(_massList[idx2]._pos._x, _massList[idx2]._pos._y, _massList[idx2]._pos._z));
	}*/


	//_debug_lines_3->update();

	
	
	// 4
	/*_debug_lines_4->clear();
	for (int b = 0; b < _massList.size(); b++)
	{
		for (int c = 0; c < _massList[b]._closestPoints3D.size(); c++)
		{
			A3DVector pt1 = _massList[b]._pos;
			A3DVector pt2 = _massList[b]._closestPoints3D[c];
			//A3DVector pt2(pt22D.x, pt22D.y, pt1._z);

			_debug_lines_4->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_debug_lines_4->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
	}
	_debug_lines_4->update();
	*/

	/*float iter = 0;
	for (int b = 0; b < _timeTriangles.size(); b++)
	{
		std::vector<A3DVector> tri;
		_tempTri3[0] = _massList[_timeTriangles[b].idx0]._pos;
		_tempTri3[1] = _massList[_timeTriangles[b].idx1]._pos;
		_tempTri3[2] = _massList[_timeTriangles[b].idx2]._pos;

		_debug_lines_4->setPoint(iter++, Ogre::Vector3(_tempTri3[0]._x, _tempTri3[0]._y, _tempTri3[0]._z));
		_debug_lines_4->setPoint(iter++, Ogre::Vector3(_tempTri3[1]._x, _tempTri3[1]._y, _tempTri3[1]._z));

		_debug_lines_4->setPoint(iter++, Ogre::Vector3(_tempTri3[1]._x, _tempTri3[1]._y, _tempTri3[1]._z));
		_debug_lines_4->setPoint(iter++, Ogre::Vector3(_tempTri3[2]._x, _tempTri3[2]._y, _tempTri3[2]._z));

		_debug_lines_4->setPoint(iter++, Ogre::Vector3(_tempTri3[2]._x, _tempTri3[2]._y, _tempTri3[2]._z));
		_debug_lines_4->setPoint(iter++, Ogre::Vector3(_tempTri3[0]._x, _tempTri3[0]._y, _tempTri3[0]._z));

	}
	_debug_lines_4->update();*/
}

void AnElement::UpdateSpringDisplayOgre3D()
{
	if (!SystemParams::_show_time_springs) { return; }

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

	_spring_lines->update();
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
void AnElement::Interp_UpdateLayerBoundaries()
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

void AnElement::CalculateLayerBoundaries_Drawing()
{	
	_per_layer_boundary_drawing.clear();
	std::vector<A3DVector> a_layer_boundary_temp(_numBoundaryPointPerLayer, A3DVector()); // temporary

	// ----- iter variables -----
	float z_iter = 0;        // negative
	int png_iter = 0;      // frames (png)
	int tube_layer_iter = 0; // layers of the tube
	
	// ----- stuff -----
	int max_num_png = SystemParams::_num_png_frame;
	float z_step = SystemParams::_upscaleFactor / ((float)max_num_png);

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

void AnElement::Interp_ResetSpringRestLengths()
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
}

void AnElement::ResetSpringRestLengths()
{
	for (int a = 0; a < _triEdges.size(); a++)
	{	
		A2DVector p1 = _massList[_triEdges[a]._index0]._pos.GetA2DVector();
		A2DVector p2 = _massList[_triEdges[a]._index1]._pos.GetA2DVector();
		_triEdges[a].SetActualOriDistance(p1.Distance(p2));
	}

	for (int a = 0; a < _auxiliaryEdges.size(); a++)
	{
		//{
		A2DVector p1 = _massList[_auxiliaryEdges[a]._index0]._pos.GetA2DVector();
		A2DVector p2 = _massList[_auxiliaryEdges[a]._index1]._pos.GetA2DVector();
		_auxiliaryEdges[a].SetActualOriDistance(p1.Distance(p2));
	}

	for (int a = 0; a < _massList.size(); a++)
	{
		_massList[a]._ori_z_pos = _massList[a]._pos._z;
	}
	
	
}

A2DVector  AnElement::Interp_ClosestPtOnALayer(A2DVector pt, int layer_idx)
{
	A2DVector closestPt;
	closestPt = UtilityFunctions::GetClosestPtOnClosedCurve(_interp_per_layer_boundary[layer_idx], pt);

	return closestPt;
}

A3DVector AnElement::ClosestPtOnATriSurface(int triIdx, A3DVector pos)
{
	A3DVector cPt = UtilityFunctions::ClosestPointOnTriangle2(pos, _timeTriangles[triIdx]._temp_1_3d, _tempTri3[1], _tempTri3[2]);
	return cPt;
}

A3DVector AnElement::ClosestPtOnTriSurfaces(std::vector<int>& triIndices, A3DVector pos)
{
	A3DVector closestPt;
	float dist = 10000000000000;
	//for (int a = 0; a < massIndices.size(); a++)
	//{
		//int idx = massIndices[a];
	for (int b = 0; b < _timeTriangles.size(); b++)
	{
		std::vector<A3DVector> tri;
		_tempTri3[0] = _massList[_timeTriangles[b].idx0]._pos;
		_tempTri3[1] = _massList[_timeTriangles[b].idx1]._pos;
		_tempTri3[2] = _massList[_timeTriangles[b].idx2]._pos;

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

void AnElement::Interp_SolveForSprings2D()
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

		if (!eForce.IsBad())
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

		if (!eForce.IsBad())
		{
			_interp_massList[idx0]._edgeForce += A3DVector(eForce.x, eForce.y, 0);	// _massList[idx0]._distToBoundary;
			_interp_massList[idx1]._edgeForce -= A3DVector(eForce.x, eForce.y, 0);	// _massList[idx1]._distToBoundary;
		}
	}
}

void AnElement::SolveForSprings2D()
{
	//float k_edge = SystemParams::_k_edge;

	A2DVector pt0;
	A2DVector pt1;
	A2DVector dir;
	A2DVector eForce;
	float dist = 0;
	float diff = 0;
	float k = 0;

	// PLEASE UNCOMMENT ME
	// PLEASE FIX ME

	for (unsigned int a = 0; a < _triEdges.size(); a++)
	{
		int idx0 = _triEdges[a]._index0;
		int idx1 = _triEdges[a]._index1;

		pt0 = _massList[idx0]._pos.GetA2DVector();
		pt1 = _massList[idx1]._pos.GetA2DVector();
				
		//if (_triEdges[a]._isLayer2Layer)
		/*{
			k = SystemParams::_k_time_edge;
			dist = pt0.Distance(pt1);
			dir = pt0.DirectionTo(pt1).Norm();
			diff = dist - _triEdges[a]._dist;
		}*/
		//else

		if (_triEdges[a]._isLayer2Layer)
		{
			k = SystemParams::_k_time_edge;
			//std::cout << ">";
			//if (_predefined_time_path)
			//{
			//	k *= 200;
			//}
			//else if (_scale > 3.0f)
			//{
			//	k *= 0.25;
			//}
		}
		else
		{
			k = SystemParams::_k_edge;

			if (_scale < 3.0f)
			{
				k *= 50;
			}
		}

		//{
			
		dir = pt0.DirectionTo(pt1);
		//dir._z = 0;
		dir = dir.Norm();
		dist = pt0.Distance(pt1);
		diff = dist - _triEdges[a]._dist;
		//}
		
		/*float signVal = 1;
		if (diff < 0) { signVal = -1; }
		eForce = (dir * k *  signVal * diff * diff);
		*/

		eForce = dir * k *  diff;

		// 2D
		/*if(_triEdges[a]._isLayer2Layer)
		{
			//eForce._z = 0;
			//eForce = eForce.Norm();
		}*/
		if (!eForce.IsBad())
		{
			_massList[idx0]._edgeForce += A3DVector(eForce.x, eForce.y, 0);	// _massList[idx0]._distToBoundary;
			_massList[idx1]._edgeForce -= A3DVector(eForce.x, eForce.y, 0);	// _massList[idx1]._distToBoundary;
		}
	}

	for (unsigned int a = 0; a < _auxiliaryEdges.size(); a++)
	{
		int idx0 = _auxiliaryEdges[a]._index0;
		int idx1 = _auxiliaryEdges[a]._index1;

		pt0 = _massList[idx0]._pos.GetA2DVector();
		pt1 = _massList[idx1]._pos.GetA2DVector();

		
		k = SystemParams::_k_edge;

		if (_scale < 3.0f)
		{
			k *= 5;
		}

		dir = pt0.DirectionTo(pt1);
		dir = dir.Norm();
		dist = pt0.Distance(pt1);
		diff = dist - _auxiliaryEdges[a]._dist;

		eForce = dir * k *  diff;

		if (!eForce.IsBad())
		{
			_massList[idx0]._edgeForce += A3DVector(eForce.x, eForce.y, 0);	// _massList[idx0]._distToBoundary;
			_massList[idx1]._edgeForce -= A3DVector(eForce.x, eForce.y, 0);	// _massList[idx1]._distToBoundary;
		}
	}
}

void AnElement::SolveForSprings3D()
{
	//float k_edge = SystemParams::_k_edge;

	A3DVector pt0;
	A3DVector pt1;
	A3DVector dir;
	A3DVector unit_dir;
	A3DVector eForce;
	float dist = 0;
	float diff = 0;
	float k = 0;

	for (unsigned int a = 0; a < _triEdges.size(); a++)
	{
		int idx0 = _triEdges[a]._index0;
		int idx1 = _triEdges[a]._index1;

		pt0 = _massList[idx0]._pos;
		pt1 = _massList[idx1]._pos;

		if (_triEdges[a]._isLayer2Layer)
		{
			k = SystemParams::_k_time_edge;
		}
		else
		{
			k = SystemParams::_k_edge;
		}

		//{

		dir = pt0.DirectionTo(pt1);
		//dir = dir.Norm();
		//dist = pt0.Distance(pt1);
		dir.GetUnitAndDist(unit_dir, dist);

		diff = dist - _triEdges[a]._dist;

		eForce = unit_dir * k *  diff;
		if (!eForce.IsBad())
		{
			_massList[idx0]._edgeForce += eForce;	// _massList[idx0]._distToBoundary;
			_massList[idx1]._edgeForce -= eForce;	// _massList[idx1]._distToBoundary;
		}
	}
}


void AnElement::UpdatePerLayerBoundaryOgre3D()
{
	//_debug_points.clear();
	_debug_lines->clear();

	float zOffset = -(SystemParams::_upscaleFactor / (SystemParams::_num_layer - 1));
	//int elem_sz = _sWorker->_element_list.size();
	//for (int a = 0; a < elem_sz; a++) // iterate element
	//{
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
}

void AnElement::UpdateClosestPtsDisplayOgre3D()
{
	//_debug_points.clear();
	_debug_lines->clear();

	/*for (int b = 0; b < _massList.size(); b++)
	{
		for (int c = 0; c < _massList[b]._closestPt_fill_sz; c++)
		{
			A3DVector pt1 = _massList[b]._pos;
			A2DVector pt22D = _massList[b]._closestPoints[c];
			A3DVector pt2(pt22D.x, pt22D.y, pt1._z);

			_debug_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_debug_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
	}*/


	//for (int i = 0; i < _debug_points.size(); i++) 
	//{
	//	_debug_lines->addPoint(_debug_points[i]);
	//}

	_debug_lines->update();
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
void  AnElement::ForceAddTriangleEdge(AnIndexedLine anEdge, int triIndex, std::vector<AnIndexedLine>& tEdges, std::vector<std::vector<int>>& e2t)
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

bool AnElement::TryToAddTriangleEdge(AnIndexedLine anEdge, int triIndex, std::vector<AnIndexedLine>& tEdges, std::vector<std::vector<int>>& e2t)
{
	int edgeIndex = FindTriangleEdge(anEdge, tEdges);
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
int AnElement::FindTriangleEdge(AnIndexedLine anEdge, std::vector<AnIndexedLine>& tEdges)
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

bool AnElement::Interp_HasOverlap()
{
	for (int a = 0; a < _interp_massList.size(); a++)
	{
		if (_interp_massList[a]._is_inside)
			return true;
	}
	return false;
}


void  AnElement::UpdateInterpMasses()
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
	
}


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
