#include "AVideoCreator.h"
#include "SystemParams.h"

AVideoCreator::AVideoCreator() : _img_scale(2.0f)
{

}

AVideoCreator::~AVideoCreator()
{

}

void AVideoCreator::Init()
{
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		CVImg img;
		img.CreateColorImage(SystemParams::_upscaleFactor * _img_scale);
		img.SetColorImageToWhite();
		_frames.push_back(img);
	}
}

void AVideoCreator::DrawLine(A2DVector pt1, A2DVector pt2, int layerIdx)
{
	_cvWrapper.DrawLine(_frames[layerIdx]._img, pt1, pt2, MyColor(0, 0, 0), 1, _img_scale);
}

void AVideoCreator::Save(std::string folderName)
{
	for (int a = 0; a < _frames.size(); a++)
	{
		std::stringstream ss;
		ss << folderName << "frame_" << a << ".png";
		_frames[a].SaveImage(ss.str());
	}
}