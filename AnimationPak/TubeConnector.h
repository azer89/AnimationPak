
#ifndef __TUBE_CONNECTOR__
#define __TUBE_CONNECTOR__

#include <vector>

struct TubeConnector
{
public:
	//int _start_1;
	//int _start_2;

	//int _end_1;
	//int _end_2;
	int _elem_1;
	int _elem_2;

	std::vector<int> _elem_1_indices;
	std::vector<int> _elem_2_indices;


	TubeConnector()
	{
		_elem_1 = -1;
		_elem_2 = -1;
		//_start_1 = -1;
		//_start_2 = -1;

		//_end_1 = -1;
		//_end_2 = -1;
	}

	~TubeConnector()
	{
	}
};

#endif 
