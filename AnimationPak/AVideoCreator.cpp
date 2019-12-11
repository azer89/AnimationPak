#include "AVideoCreator.h"
#include "SystemParams.h"

AVideoCreator::AVideoCreator() : _img_scale(4.0f)
{

}

AVideoCreator::~AVideoCreator()
{

}

void AVideoCreator::Init(int numInterpolation)
{
	_num_frame = numInterpolation;
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

void AVideoCreator::DrawFilledArt(std::vector<std::vector<A2DVector>> arts, MyColor color, int frameIdx)
{
	for (int a = arts.size() - 1; a >= 0; a--) // backward
	{
		_cvWrapper.DrawFilledPoly(_frames[frameIdx], arts[a], color, _img_scale);
	}
}

void  AVideoCreator::DrawFilledArt(std::vector<std::vector<A2DVector>> arts,
									std::vector <MyColor> b_colors,
									std::vector <MyColor> f_colors,
									int frameIdx)
{
	for (int a = arts.size() - 1; a >= 0; a--) // backward
	{
		// fill
		if (b_colors[a].IsValid())
		{
			_cvWrapper.DrawFilledPoly(_frames[frameIdx], arts[a], b_colors[a], _img_scale);
		}

		// stroke
		if (f_colors[a].IsValid())
		{
			_cvWrapper.DrawPolyOnCVImage(_frames[frameIdx]._img, arts[a], f_colors[a], true, 1.0f, _img_scale);
		}
		//else
		//{
		//	_cvWrapper.DrawPolyOnCVImage(_frames[frameIdx]._img, arts[a], MyColor(0, 0, 0), true, 1.0f, _img_scale);
		//}
	}
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