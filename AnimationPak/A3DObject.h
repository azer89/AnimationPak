#ifndef __A_3D_Object__
#define __A_3D_Object__

class A3DObject {
public:
	float _x;
	float _y;
	float _z;
	int   _info1;  // which element
	int   _info2;  // which mass



	A3DObject(float x, float y, float z, int info1, int info2) :
		_x(x),
		_y(y),
		_z(z),
		_info1(info1),
		_info2(info2)//,
					 //_nx(-1000),
					 //_ny(-1000)
	{
	}

	A3DObject(float x, float y, float z, int info1) :
		_x(x),
		_y(y),
		_z(z),
		_info1(info1),
		_info2(-1)//,
				  //_nx(-1000),
				  //_ny(-1000)
	{
	}

	A3DObject(float x, float y, float z) :
		_x(x),
		_y(y),
		_z(z),
		_info1(-1),
		_info2(-1)//,
				  //_nx(-1000),
				  //_ny(-1000)
	{
	}

	//bool HasNormalVector()
	//{
	//	return (this->_nx > -500 && this->_ny > -500);
	//}
};

#endif
