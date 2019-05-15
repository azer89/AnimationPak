#include "AVideoCreator.h"
#include "SystemParams.h"

AVideoCreator::AVideoCreator() : _img_scale(2.0f)
{

}

AVideoCreator::~AVideoCreator()
{

}

void AVideoCreator::Init(int numInterpolation)
{
	_num_frame = numInterpolation * SystemParams::_num_layer;
	for (int a = 0; a < _num_frame; a++)
	{
		CVImg img;
		img.CreateColorImage(SystemParams::_upscaleFactor * _img_scale);
		img.SetColorImageToWhite();
		_frames.push_back(img);
	}
}

void AVideoCreator::DrawRedCircle(int frameIdx)
{
	_cvWrapper.DrawCircle(_frames[frameIdx]._img, A2DVector(20, 20), MyColor(255, 0, 0), 10);
}

void AVideoCreator::DrawLine(A2DVector pt1, A2DVector pt2, MyColor color, int frameIdx)
{
	_cvWrapper.DrawLine(_frames[frameIdx]._img, pt1, pt2, color, _img_scale, _img_scale);
}

void AVideoCreator::ClearFrames()
{
	for (int a = 0; a < _frames.size(); a++)
	{
		_frames[a].SetColorImageToWhite();
	}
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