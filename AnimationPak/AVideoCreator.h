#ifndef AVIDEOCREATOR_H
#define AVIDEOCREATOR_H

#include <vector>
#include <string>

#include "OpenCVWrapper.h"
#include "A2DVector.h"

class AVideoCreator
{
public:
	AVideoCreator();
	~AVideoCreator();

	void Init(int numInterpolation = 1);
	void DrawLine(A2DVector pt1, A2DVector pt2, MyColor color, int frameIdx);
	void DrawRedCircle(int frameIdx);
	void Save(std::string folderName);

private:
	float _img_scale;
	int _num_frame;
	std::vector<CVImg> _frames;
	OpenCVWrapper _cvWrapper;

};

#endif
