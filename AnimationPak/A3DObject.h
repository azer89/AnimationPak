#ifndef __A_3D_Object__
#define __A_3D_Object__

class A3DObject {
public:
	// for octree it's left hand rule
	// https_//ogrecave.github.io/ogre/api/1.11/tut__first_scene.html
	float _x;
	float _y;
	float _z;
	int   _info1;  // which element
	int   _info2;  // which mass or which AnElement::_timeTriangles???
	int   _info3;  // not used

	// only for octree, not uniform grid
	//float _x_width;
	//float _y_width;
	//float _z_width;
	float _x_max;
	float _y_max;
	float _z_max;

	A3DObject(float x, float y, float z, int info1, int info2, int info3) :
		_x(x),
		_y(y),
		_z(z),
		_info1(info1),
		_info2(info2),
		_info3(info3)//,
					 //_nx(-1000),
					 //_ny(-1000)
	{
		// only for octree, not uniform grid
		_x_max = _x;
		_y_max = _y;
		_z_max = _z;
		//_x_width = 0;
		//_y_width = 0;
		//_z_width = 0;
	}


	A3DObject(float x, float y, float z, int info1, int info2) :
		_x(x),
		_y(y),
		_z(z),
		_info1(info1),
		_info2(info2),
		_info3(-1)//,
					 //_nx(-1000),
					 //_ny(-1000)
	{
		// only for octree, not uniform grid
		_x_max = _x;
		_y_max = _y;
		_z_max = _z;
		//_x_width = 0;
		//_y_width = 0;
		//_z_width = 0;
	}

	A3DObject(float x, float y, float z, int info1) :
		_x(x),
		_y(y),
		_z(z),
		_info1(info1),
		_info2(-1),
		_info3(-1)//,
				  //_nx(-1000),
				  //_ny(-1000)
	{
		// only for octree, not uniform grid
		_x_max = _x;
		_y_max = _y;
		_z_max = _z;
		//_x_width = 0;
		//_y_width = 0;
		//_z_width = 0;	
	}

	A3DObject(float x, float y, float z) :
		_x(x),
		_y(y),
		_z(z),
		_info1(-1),
		_info2(-1),
		_info3(-1)//,
				  //_nx(-1000),
				  //_ny(-1000)
	{
		// only for octree, not uniform grid
		_x_max = _x;
		_y_max = _y;
		_z_max = _z;
		//_x_width = 0;
		//_y_width = 0;
		//_z_width = 0;
	}

	void SetMaxVals(float x_max, float y_max, float z_max)
	{
		_x_max = x_max;
		_y_max = y_max;
		_z_max = z_max;
	}

	//bool HasNormalVector()
	//{
	//	return (this->_nx > -500 && this->_ny > -500);
	//}
};

#endif
