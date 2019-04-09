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

	void Init();
	void DrawLine(A2DVector pt1, A2DVector pt2, int layerIdx);
	void Save(std::string folderName);

private:
	float _img_scale;
	std::vector<CVImg> _frames;
	OpenCVWrapper _cvWrapper;

};

#endif
