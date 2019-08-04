
/* ---------- ShapeRadiusMatching V2  ---------- */

#include "OpenCVWrapper.h"
#include "A2DVector.h"
#include "A3DVector.h"

//#include "ARectangle.h"
//#include "UtilityFunctions.h"
#include "NANOFLANNWrapper2D.h"

//#include "ClipperWrapper.h"

#include "SystemParams.h"

#include <iostream>
#include <cmath>

// OpenGL
//#include "glew.h"
//#include "freeglut.h"

//#include "imgui.h"
//#include "imgui_impl_glut.h"


//MyColor MyColor::_black = MyColor(0, 0, 0);
//MyColor MyColor::_white = MyColor(255, 255, 255);

/*================================================================================
================================================================================*/
OpenCVWrapper::OpenCVWrapper()
{
	this->_rng = cv::RNG(12345);
}

/*================================================================================
================================================================================*/
OpenCVWrapper::~OpenCVWrapper()
{
}

/*================================================================================
================================================================================*/
MyColor OpenCVWrapper::GetRandomColor()
{
	return MyColor(_rng.uniform(0, 255), _rng.uniform(0, 255), _rng.uniform(0, 255));
}

/*================================================================================
================================================================================*/
void OpenCVWrapper::CreateImage(std::string imageName, int width, int height, int imgType)
{
	cv::Mat newImg = cv::Mat::zeros(width, height, imgType);
	_images[imageName] = newImg;
}

/*================================================================================
================================================================================*/
void OpenCVWrapper::ShowImage(std::string imageName)
{
	cv::Mat img = _images[imageName];
	//cv::namedWindow(imageName, CV_WINDOW_AUTOSIZE);
	cv::imshow(imageName, img);
}


/*================================================================================
================================================================================*/
void OpenCVWrapper::SaveImage(std::string imageName, std::string filePath)
{
	cv::Mat img = _images[imageName];
	cv::imwrite(filePath, img);
}

/*================================================================================
BGR_255
================================================================================*/
MyColor OpenCVWrapper::GetColor(std::string imageName, int x, int y)
{
	// flipped?
	cv::Mat img = _images[imageName];
	int r = img.at<cv::Vec3b>(y, x)[0];
	int g = img.at<cv::Vec3b>(y, x)[1];
	int b = img.at<cv::Vec3b>(y, x)[2];
	return MyColor(r, g, b);
}

/*================================================================================
// get centroid
// https_//stackoverflow.com/questions/9074202/opencv-2-centroid
================================================================================*/
A2DVector OpenCVWrapper::GetPolygonCentroid(const std::vector<A2DVector>& contour)
{
	std::vector<cv::Point2f> cvContour = ConvertList<A2DVector, cv::Point2f>(contour);
	cv::Moments mm = cv::moments(cvContour, false);
	return A2DVector(mm.m10 / mm.m00, mm.m01 / mm.m00);
}

/*================================================================================
================================================================================*/
std::vector<A2DVector> OpenCVWrapper::GetContour(CVImg img)
{
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(img.GetCVImage(), contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	//std::cout << img._img.cols << " - " << img._img.rows << "\n";
	int longestIdx = GetLongestContourIndex(contours);
	return ConvertList<cv::Point, A2DVector>(contours[longestIdx]);
}

/*================================================================================
================================================================================*/
std::vector<std::vector<A2DVector>> OpenCVWrapper::GetContours(CVImg img, int blurSize)
{
	std::vector<std::vector<A2DVector>> myContours;
	std::vector<std::vector<cv::Point> > cvContours;
	std::vector<cv::Vec4i> hierarchy;

	cv::Mat gray_img = img.GetCVImage();

	if (blurSize > 0)
	{
		cv::blur(gray_img, gray_img, cv::Size(blurSize, blurSize));
	}

	findContours(gray_img, cvContours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	for (int a = 0; a < cvContours.size(); a++)
	{
		std::vector<A2DVector> ctr = ConvertList<cv::Point, A2DVector>(cvContours[a]);
		myContours.push_back(ctr);
	}

	return myContours;
}


/*================================================================================
BGR_255
================================================================================*/
void OpenCVWrapper::SetPixel(std::string imageName, int x, int y, MyColor col)
{
	// flipped?
	cv::Mat img = _images[imageName];
	img.at<cv::Vec3b>(y, x)[0] = col._b;
	img.at<cv::Vec3b>(y, x)[1] = col._g;
	img.at<cv::Vec3b>(y, x)[2] = col._r;
}

/*================================================================================
================================================================================*/
int OpenCVWrapper::GetNumColumns(std::string imageName)
{
	cv::Mat img = _images[imageName];
	return img.cols;
}

/*================================================================================
================================================================================*/
int OpenCVWrapper::GetNumRows(std::string imageName)
{
	cv::Mat img = _images[imageName];
	return img.rows;
}

/*================================================================================
================================================================================*/
void OpenCVWrapper::WaitKey()
{
	cv::waitKey();
}

/*================================================================================
================================================================================*/
void OpenCVWrapper::SetWhite(std::string imageName)
{
	cv::Mat img = _images[imageName];
	img = cv::Scalar(255, 255, 255);
}

/*================================================================================
================================================================================*/
void OpenCVWrapper::SetBlack(std::string imageName)
{
	cv::Mat img = _images[imageName];
	img = cv::Scalar(0, 0, 0);
}

/*================================================================================
================================================================================*/
CVImg OpenCVWrapper::ConvertToBW(std::string imageName)
{
	//std::cout << "ConvertToBW\n";
	float sz = SystemParams::_upscaleFactor;

	cv::Mat img = _images[imageName];
	CVImg bwImg;
	//std::cout << img.rows << " " << img.cols << "\n";
	bwImg.CreateIntegerImage(sz, sz);

	for (int x = 0; x < sz; x++)
	{
		for (int y = 0; y < sz; y++)
		{
			if (img.at<cv::Vec3b>(y, x)[0] == 0 && img.at<cv::Vec3b>(y, x)[1] == 0 && img.at<cv::Vec3b>(y, x)[2] == 0)
			{
				bwImg.SetIntegerPixel(x, y, 0);
			}
			else
			{
				bwImg.SetIntegerPixel(x, y, 1);
			}
		}
	}
	return bwImg;
}

/*================================================================================
================================================================================*/
/*std::vector<cv::Point2f> OpenCVWrapper::ConvertAVectorToCvPoint2f(std::vector<AVector> contour)
{
std::vector<cv::Point2f> cvContour;
for (int a = 0; a < contour.size(); a++)
{ cvContour.push_back(cv::Point2f(contour[a].x, contour[a].y)); }
return cvContour;
}*/

/*================================================================================
================================================================================*/
template <typename T, typename U>
std::vector<U> OpenCVWrapper::ConvertList(const std::vector<T>& contour1)
{
	std::vector<U> contour2;
	for (int a = 0; a < contour1.size(); a++)
	{
		contour2.push_back(U(contour1[a].x, contour1[a].y));
	}
	return contour2;
}

// Point   --> AVector
template std::vector<A2DVector>     OpenCVWrapper::ConvertList(const std::vector<cv::Point>& contour1);

// AVector --> Point
template std::vector<cv::Point>   OpenCVWrapper::ConvertList(const std::vector<A2DVector>& contour1);

// Point2f --> AVector
template std::vector<A2DVector>     OpenCVWrapper::ConvertList(const std::vector<cv::Point2f>& contour1);

// AVector --> Point2f
template std::vector<cv::Point2f> OpenCVWrapper::ConvertList(const std::vector<A2DVector>& contour1);

/*================================================================================
================================================================================*/
/*std::vector<cv::Point> OpenCVWrapper::ConvertAVectorToCvPoint(std::vector<AVector> contour)
{
std::vector<cv::Point> cvContour;
for (int a = 0; a < contour.size(); a++)
{ cvContour.push_back(cv::Point(contour[a].x, contour[a].y)); }
return cvContour;
}*/

/*================================================================================
================================================================================*/
template <typename T>
float OpenCVWrapper::PointPolygonTest(std::vector<T> shape_contours, T pt)
{
	std::vector<cv::Point2f> new_contours;
	for (size_t a = 0; a < shape_contours.size(); a++)
	{
		new_contours.push_back(cv::Point2f((shape_contours[a].x), (shape_contours[a].y)));
	}

	return cv::pointPolygonTest(new_contours, cv::Point2f(pt.x, pt.y), true);
}

template
float OpenCVWrapper::PointPolygonTest(std::vector<A2DVector> shape_contours, A2DVector pt);

/*================================================================================
================================================================================*/
CVImg OpenCVWrapper::CalculateSignedDistance(const std::vector<A2DVector>& contour)
{
	//std::vector<cv::Point> cvContour = ConvertAVectorToCvPoint(contour);
	std::vector<cv::Point> cvContour = ConvertList<A2DVector, cv::Point>(contour);

	int sz = SystemParams::_upscaleFactor;

	CVImg rawDist;
	rawDist.CreateFloatImage(sz);

	for (int x = 0; x < sz; x++)
	{
		for (int y = 0; y < sz; y++)
		{
			float d = cv::pointPolygonTest(cvContour, cv::Point2f(x, y), true);
			rawDist.SetFloatPixel(x, y, -d); // because I want positive value outside
		}
	}

	return rawDist;
}

/*================================================================================
================================================================================*/
CVImg OpenCVWrapper::CalculateSignedDistance(const std::vector<A2DVector>& contour, CVImg aMask)
{
	//std::vector<cv::Point> cvContour = ConvertAVectorToCvPoint(contour);
	std::vector<cv::Point> cvContour = ConvertList<A2DVector, cv::Point>(contour);

	int sz = SystemParams::_upscaleFactor;

	CVImg rawDist;
	rawDist.CreateFloatImage(sz);

	for (int x = 0; x < sz; x++)
	{
		for (int y = 0; y < sz; y++)
		{
			float d = sz * sz;
			//if (UtilityFunctions::InsidePolygon(boundary, AVector(x, y)))
			if (aMask.GetIntegerValue(x, y) > 0)
			{
				// because I want positive value outside
				d = -cv::pointPolygonTest(cvContour, cv::Point2f(x, y), true);
			}
			rawDist.SetFloatPixel(x, y, d);
		}
	}
	return rawDist;
}

/*================================================================================
================================================================================*/
float OpenCVWrapper::GetSignedDistance(const std::vector<cv::Point>& contour, int x, int y)
{
	// https_:_//github.com/opencv/opencv/blob/master/modules/imgproc/src/geometry.cpp
	return -cv::pointPolygonTest(contour, cv::Point2f(x, y), true);
}

/*================================================================================
================================================================================*/
float OpenCVWrapper::GetSignedDistance(const std::vector<std::vector<cv::Point>>& contours, int x, int y)
{
	//return -cv::pointPolygonTest(contour, cv::Point2f(x, y), true);
	float dist = std::numeric_limits<float>::max();
	for (int a = 0; a < contours.size(); a++)
	{
		float d = GetSignedDistance(contours[a], x, y);
		if (d < dist) { dist = d; }
	}
	return dist;
}

/*================================================================================
================================================================================*/
A2DVector OpenCVWrapper::GetCenter(const std::vector<A3DVector>& polygon)
{
	//std::vector<cv::Point2f> cvPolygon = ConvertList<A3DVector, cv::Point2f>(polygon);
	
	std::vector<cv::Point2f> cvPolygon;
	for (int a = 0; a < polygon.size(); a++)
		{ cvPolygon.push_back(cv::Point2f(polygon[a]._x, polygon[a]._y)); }
	
	cv::Moments mu = cv::moments(cvPolygon, false);
	return A2DVector(mu.m10 / mu.m00, mu.m01 / mu.m00);
}

/*================================================================================
================================================================================*/
A2DVector OpenCVWrapper::GetCenter(const std::vector<A2DVector>& polygon)
{
	std::vector<cv::Point2f> cvPolygon = ConvertList<A2DVector, cv::Point2f>(polygon);
	cv::Moments mu = cv::moments(cvPolygon, false);
	return A2DVector(mu.m10 / mu.m00, mu.m01 / mu.m00);
}

/*================================================================================
================================================================================*/
std::vector<A2DVector> OpenCVWrapper::GetConvexHull(std::vector<A2DVector> polygon)
{
	std::vector<cv::Point2f> cvPolygon = ConvertList<A2DVector, cv::Point2f>(polygon);
	std::vector<cv::Point2f> cvHull;
	cv::convexHull(cv::Mat(cvPolygon), cvHull, false);

	return ConvertList<cv::Point2f, A2DVector>(cvHull);
}

/*================================================================================
================================================================================*/
template <typename T>
void OpenCVWrapper::DrawPolysOnCVImage(cv::Mat img,
	const std::vector<std::vector<T>>& shape_contours,
	MyColor color,
	bool isClosed,
	float thickness,
	float scale,
	float xOffset,
	float yOffset)
{
	for (unsigned int a = 0; a < shape_contours.size(); a++)
	{
		DrawPolyOnCVImage(img, shape_contours[a], color, isClosed, thickness, scale, xOffset, yOffset);
	}
}

template
void OpenCVWrapper::DrawPolysOnCVImage(cv::Mat img,
	const std::vector<std::vector<A2DVector>>& shape_contours,
	MyColor color,
	bool isClosed,
	float thickness,
	float scale,
	float xOffset,
	float yOffset);

/*================================================================================
================================================================================*/
template <typename T>
void OpenCVWrapper::DrawPolyOnCVImage(cv::Mat img,
	const std::vector<T>& shape_contours,
	MyColor color,
	bool isClosed,
	float thickness,
	float scale,
	float xOffset,
	float yOffset)
{
	std::vector<cv::Point2f> new_contours;
	for (size_t b = 0; b < shape_contours.size(); b++)
	{
		new_contours.push_back(cv::Point2f((shape_contours[b].x * scale + xOffset), (shape_contours[b].y * scale + yOffset)));
	}
	for (size_t b = 0; b < new_contours.size() - 1; b++)
	{
		cv::line(img, new_contours[b], new_contours[b + 1], cv::Scalar(color._b, color._g, color._r), thickness);
	}
	if (isClosed)
	{
		cv::line(img, new_contours[new_contours.size() - 1], new_contours[0], cv::Scalar(color._b, color._g, color._r), thickness);
	}
}

template
void OpenCVWrapper::DrawPolyOnCVImage(cv::Mat img,
	const std::vector<A2DVector>& shape_contours,
	MyColor color,
	bool isClosed,
	float thickness,
	float scale,
	float xOffset,
	float yOffset);

template
void OpenCVWrapper::DrawPolyOnCVImage(cv::Mat img,
	const std::vector<cv::Point2f>& shape_contours,
	MyColor color,
	bool isClosed,
	float thickness,
	float scale,
	float xOffset,
	float yOffset);


/*================================================================================
================================================================================*/
template <typename T>
void OpenCVWrapper::DrawPoly(std::string imageName,
	const std::vector<T>& shape_contours,
	MyColor color,
	bool isClosed,
	float thickness,
	float scale,
	float xOffset,
	float yOffset)
{
	cv::Mat drawing = _images[imageName];

	std::vector<cv::Point2f> new_contours;
	for (size_t b = 0; b < shape_contours.size(); b++)
	{
		new_contours.push_back(cv::Point2f((shape_contours[b].x * scale + xOffset), (shape_contours[b].y * scale + yOffset)));
	}
	for (size_t b = 0; b < new_contours.size() - 1; b++)
	{
		cv::line(drawing, new_contours[b], new_contours[b + 1], cv::Scalar(color._b, color._g, color._r), thickness);
	}
	if (isClosed)
	{
		cv::line(drawing, new_contours[new_contours.size() - 1], new_contours[0], cv::Scalar(color._b, color._g, color._r), thickness);
	}
}

template
void OpenCVWrapper::DrawPoly(std::string imageName,
	const std::vector<A2DVector>& shape_contours,
	MyColor color,
	bool isClosed,
	float thickness,
	float scale,
	float xOffset,
	float yOffset);

/*================================================================================
================================================================================*/
template <typename T>
void OpenCVWrapper::DrawPolys(std::string imageName,
	const std::vector<std::vector<T>>& shape_contours,
	MyColor color,
	bool isClosed,
	float thickness,
	float scale,
	float xOffset,
	float yOffset)
{
	for (int a = 0; a < shape_contours.size(); a++)
	{
		DrawPoly(imageName,
			shape_contours[a],
			color,
			isClosed,
			thickness,
			scale,
			xOffset,
			yOffset);
	}
}

template
void OpenCVWrapper::DrawPolys(std::string imageName,
	const std::vector<std::vector<A2DVector>>& shape_contours,
	MyColor color,
	bool isClosed,
	float thickness,
	float scale,
	float xOffset,
	float yOffset);




/*================================================================================
================================================================================*/
template <typename T>
void OpenCVWrapper::DrawFilledPoly(std::string imageName,
	const std::vector<T>& shape_contours,
	MyColor color,
	float scale,
	float xOffset,
	float yOffset)
{
	cv::Mat drawing = _images[imageName];

	std::vector<cv::Point> new_contours;
	for (size_t b = 0; b < shape_contours.size(); b++)
	{
		new_contours.push_back(cv::Point((int)(shape_contours[b].x * scale + xOffset), int(shape_contours[b].y * scale + yOffset)));
	}
	std::vector<std::vector<cv::Point>> contours;
	contours.push_back(new_contours);
	cv::fillPoly(drawing, contours, cv::Scalar(color._b, color._g, color._r));
}

template
void OpenCVWrapper::DrawFilledPoly(std::string imageName,
	const std::vector<A2DVector>& shape_contours,
	MyColor color,
	float scale,
	float xOffset,
	float yOffset);

/*================================================================================
================================================================================*/
template <typename T>
void OpenCVWrapper::DrawFilledPoly(CVImg img,
	const std::vector<T>& shape_contours,
	MyColor color,
	float scale,
	float xOffset,
	float yOffset)
{
	std::vector<cv::Point> new_contours;
	for (size_t b = 0; b < shape_contours.size(); b++)
	{
		new_contours.push_back(cv::Point((int)(shape_contours[b].x * scale + xOffset), int(shape_contours[b].y * scale + yOffset)));
	}
	std::vector<std::vector<cv::Point>> contours;
	contours.push_back(new_contours);
	cv::fillPoly(img._img, contours, cv::Scalar(color._b, color._g, color._r));
}

template
void OpenCVWrapper::DrawFilledPoly(CVImg img,
	const std::vector<A2DVector>& shape_contours,
	MyColor color,
	float scale,
	float xOffset,
	float yOffset);



/*================================================================================
================================================================================*/
template <typename T>
void OpenCVWrapper::DrawFilledPolyInt(CVImg& img,
	const std::vector<T>& shape_contours,
	int val,
	float scale,
	float xOffset,
	float yOffset)
{
	std::vector<cv::Point> new_contours;
	for (size_t b = 0; b < shape_contours.size(); b++)
	{
		new_contours.push_back(cv::Point((int)(shape_contours[b].x * scale + xOffset), int(shape_contours[b].y * scale + yOffset)));
	}
	std::vector<std::vector<cv::Point>> contours;
	contours.push_back(new_contours);
	cv::fillPoly(img._img, contours, val);
}

template
void OpenCVWrapper::DrawFilledPolyInt(CVImg& img,
	const std::vector<A2DVector>& shape_contours,
	int val,
	float scale,
	float xOffset,
	float yOffset);

template
void OpenCVWrapper::DrawFilledPolyInt(CVImg& img,
	const std::vector<cv::Point2f>& shape_contours,
	int val,
	float scale,
	float xOffset,
	float yOffset);

/*================================================================================
================================================================================*/
template <typename T>
void OpenCVWrapper::DrawPolyInt(CVImg& img,
	const std::vector<T>& shape_contours,
	int val,
	bool isClosed,
	float thickness,
	float scale,
	float xOffset,
	float yOffset)
{
	std::vector<cv::Point2f> new_contours;
	for (size_t b = 0; b < shape_contours.size(); b++)
	{
		new_contours.push_back(cv::Point2f((shape_contours[b].x * scale + xOffset), (shape_contours[b].y * scale + yOffset)));
	}
	for (size_t b = 0; b < new_contours.size() - 1; b++)
	{
		cv::line(img._img, new_contours[b], new_contours[b + 1], val, thickness);
	}
	if (isClosed)
	{
		cv::line(img._img, new_contours[new_contours.size() - 1], new_contours[0], val, thickness);
	}
}

template
void OpenCVWrapper::DrawPolyInt(CVImg& img,
	const std::vector<A2DVector>& shape_contours,
	int val,
	bool isClosed,
	float thickness,
	float scale,
	float xOffset,
	float yOffset);

/*================================================================================
================================================================================*/

// ---------- draw ----------
template <typename T>
void OpenCVWrapper::DrawLineInt(CVImg& img, T pt1, T pt2, int val, int thickness, float scale)
{
	cv::line(img._img, cv::Point(pt1.x, pt1.y) * scale, cv::Point(pt2.x, pt2.y) * scale, val, thickness);
}

template
void OpenCVWrapper::DrawLineInt(CVImg& img, A2DVector pt1, A2DVector pt2, int val, int thickness, float scale);

/*================================================================================
================================================================================*/
template <typename T>
void OpenCVWrapper::DrawLine(std::string imageName, T pt1, T pt2, MyColor color, int thickness, float scale)
{
	cv::Mat drawing = _images[imageName];
	cv::line(drawing, cv::Point(pt1.x, pt1.y) * scale, cv::Point(pt2.x, pt2.y) * scale, cv::Scalar(color._b, color._g, color._r), thickness);
}

template
void OpenCVWrapper::DrawLine(std::string imageName, A2DVector pt1, A2DVector pt2, MyColor color, int thickness, float scale);

/*================================================================================
================================================================================*/

template <typename T>
void OpenCVWrapper::DrawLine(cv::Mat img, T pt1, T pt2, MyColor color, int thickness, float scale)
{
	cv::line(img, cv::Point(pt1.x, pt1.y) * scale, cv::Point(pt2.x, pt2.y) * scale, cv::Scalar(color._b, color._g, color._r), thickness);
}

template
void OpenCVWrapper::DrawLine(cv::Mat img, A2DVector pt1, A2DVector pt2, MyColor color, int thickness, float scale);

/*================================================================================
================================================================================*/
template <typename T>
int OpenCVWrapper::GetLongestContourIndex(std::vector<std::vector<T>> contours)
{
	if (contours.size() == 1) { return 0; }
	else if (contours.size() == 0) { std::cout << "GetLongestContour function says: error no contour\n"; }

	int idx = -1;
	int sz = -1;
	for (int a = 0; a < contours.size(); a++)
	{
		int s = contours[a].size();
		if (s > sz)
		{
			sz = s;
			idx = a;
		}
	}
	return idx;
}

template int OpenCVWrapper::GetLongestContourIndex<cv::Point>(std::vector<std::vector<cv::Point>> contours); // opencv
template int OpenCVWrapper::GetLongestContourIndex<A2DVector>(std::vector<std::vector<A2DVector>> contours); // AVector

																										 /*================================================================================
																										 ================================================================================*/
void OpenCVWrapper::PutText(std::string imageName, std::string text, A2DVector pos, MyColor col, float scale, float thickness)
{
	cv::Mat img = _images[imageName];
	cv::putText(img, text, cv::Point(pos.x, pos.y), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(col._b, col._g, col._r), thickness);
}

void OpenCVWrapper::PutText(cv::Mat img, std::string text, A2DVector pos, MyColor col, float scale, float thickness)
{
	cv::putText(img, text, cv::Point(pos.x, pos.y), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(col._b, col._g, col._r), thickness);
}

/*================================================================================
================================================================================*/
// I'm trying to be fast
/*CVImg OpenCVWrapper::GetDistanceMapFast(int numRegion,
	const std::vector<ArtData>& artDataArray,
	CVImg artDataMask)
{
	//clock_t begin_time = clock();
	std::cout << "OpenCVWrapper::GetDistanceMapFast: function not implemented!!!\n";

	int sz = SystemParams::_upscaleFactor;
	CVImg distanceMat;

	return distanceMat;
}*/

/*================================================================================
================================================================================*/
// bottlenect, this function is really slow
/*CVImg OpenCVWrapper::GetDistanceMapSlow(int numRegion,
	VFRegion* aRegion,
	const std::vector<ArtData>& artDataArray,
	const std::vector<AVector>& boundary,
	const std::vector<std::vector<AVector>>& focalBoundaries)
{
	//clock_t begin_time = clock();

	int sz = SystemParams::_upscaleFactor;
	CVImg distanceMat;

	std::stringstream ss1;
	ss1 << SystemParams::_image_folder << "\\segmentation\\" << SystemParams::_artName << "_region_" << numRegion << ".floatmap";

	std::stringstream ss2;
	ss2 << SystemParams::_image_folder << "\\segmentation\\" << SystemParams::_artName << "_region_" << numRegion << ".png";

	//if (_recalculate_distance_map) // calculate
	//{
	distanceMat.CreateFloatImage(sz);

	// all ornaments (float-integer warning)
	std::vector<std::vector<cv::Point>> allCVOrnaments;
	for (int a = 0; a < artDataArray.size(); a++)
	{
		if (artDataArray[a]._regionNumber == numRegion)
		{
			std::vector<std::vector<cv::Point>> cvContours = GetCVContours(artDataArray[a]._boundaries);
			allCVOrnaments.insert(allCVOrnaments.end(), cvContours.begin(), cvContours.end());
		}
	}

	for (int y = 0; y < sz; y++)
	{
		for (int x = 0; x < sz; x++)
		{
			if (UtilityFunctions::InsidePolygon(boundary, x, y) && !UtilityFunctions::InsidePolygons(focalBoundaries, x, y))
			{
				float dist = GetSignedDistance(allCVOrnaments, x, y);
				distanceMat.SetFloatPixel(x, y, dist);
			}
		}
	}



	distanceMat.SaveDistanceImage(ss2.str());

	//std::cout << "GetDistanceMapSlow time: " << float(clock() - begin_time) / CLOCKS_PER_SEC << "\n";

	return distanceMat;
}*/

// ---------- from ShapeRadiusMatching ----------
std::vector<std::vector<cv::Point>> OpenCVWrapper::GetCVContours(const std::vector<std::vector<A2DVector>>& contours)
{
	// OpenCV Code
	std::vector<std::vector<cv::Point>> cvContours;
	for (int a = 0; a < contours.size(); a++)
	{
		std::vector<cv::Point> cvContour = ConvertList<A2DVector, cv::Point>(contours[a]);
		cvContours.push_back(cvContour);
	}

	return cvContours;
}



/*CVImg OpenCVWrapper::CreateDijkstraMask(int blobNumber,
	std::vector<AVector> combinedBoundary,
	std::vector<ArtData> artDataArray)
{
	//float lineWidth = 1;

	CVImg bwImg;
	bwImg.CreateIntegerImage(SystemParams::_upscaleFactor);

	// white (1) inside the boundary
	DrawFilledPolyInt(bwImg, combinedBoundary, 1, true, 1);

	// the interior of the ornaments should be black
	for (int a = 0; a < artDataArray.size(); a++)
	{
		if (artDataArray[a]._blobNumber == blobNumber) { continue; }

		std::vector<std::vector<AVector>> boundaries = artDataArray[a]._boundaries;

		for (int b = 0; b < boundaries.size(); b++)
		{
			//std::vector<std::vector<AVector>> offPolys = ClipperWrapper::MiterOffsettingP(boundaries[b], lineWidth / 2.0f, 7);
			//for (int c = 0; c < offPolys.size(); c++)
			//{
			DrawFilledPolyInt(bwImg, boundaries[b], 0);
			//}
		}
	}

	return bwImg;
}*/


void OpenCVWrapper::Triangulate(std::vector<AnIdxTriangle>& myTriangles,
	std::vector<AnIndexedLine>& negSpaceEdges,
	const std::vector<A2DVector>& randomPoints,
	const std::vector<A2DVector>& boundary,
	float img_length/*,
	const std::vector<std::vector<A2DVector>>& arts*/)
{
	std::vector<cv::Point2f> cvBoundary = ConvertList<A2DVector, cv::Point2f>(boundary);


	std::vector<int> indices;
	NANOFLANNWrapper2D* knn = new NANOFLANNWrapper2D();
	knn->_leaf_max_size = 4;
	knn->SetPointData(randomPoints);
	knn->CreatePointKDTree();

	//std::cout << "img_length: " << img_length << "\n";

	cv::Rect rect(0, 0, img_length, img_length);
	cv::Subdiv2D subdiv(rect);

	for (int a = 0; a < randomPoints.size(); a++)
	{
		subdiv.insert(cv::Point2f(randomPoints[a].x, randomPoints[a].y));
	}


	float eps = 1e-1;

	//std::vector<AnIdxTriangle> myTriangles;
	std::vector<cv::Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	for (size_t i = 0; i < triangleList.size(); i++)
	{
		A2DVector pt1(triangleList[i][0], triangleList[i][1]);
		A2DVector pt2(triangleList[i][2], triangleList[i][3]);
		A2DVector pt3(triangleList[i][4], triangleList[i][5]);

		int idx1 = knn->GetClosestIndices(pt1, 1)[0];
		int idx2 = knn->GetClosestIndices(pt2, 1)[0];
		int idx3 = knn->GetClosestIndices(pt3, 1)[0];

		A2DVector randPt1 = randomPoints[idx1];
		A2DVector randPt2 = randomPoints[idx2];
		A2DVector randPt3 = randomPoints[idx3];
		if (randPt1.Distance(pt1) > eps) { continue; }
		if (randPt2.Distance(pt2) > eps) { continue; }
		if (randPt3.Distance(pt3) > eps) { continue; }


		A2DVector centerPt = (pt1 + pt2 + pt3) / 3.0f;
		if (cv::pointPolygonTest(cvBoundary, cv::Point2f(centerPt.x, centerPt.y), true) < 0)
		{
			// 1 & 2
			if (std::abs(idx1 - idx2) > 2)
			{
				negSpaceEdges.push_back(AnIndexedLine(idx1, idx2));
			}

			// 2 & 3
			if (std::abs(idx2 - idx3) > 2)
			{
				negSpaceEdges.push_back(AnIndexedLine(idx2, idx3));
			}

			// 3 & 1
			if (std::abs(idx3 - idx1) > 2)
			{
				negSpaceEdges.push_back(AnIndexedLine(idx3, idx1));
			}

			continue;
		}


		//std::vector<A2DVector> triPts = { pt1, pt2, pt3 };
		//if (ClipperWrapper::IsClockwise(triPts))
		//{
		AnIdxTriangle tri(idx1, idx2, idx3);
		myTriangles.push_back(tri);
		//}
		/*else
		{
			std::cout << "flip\n";
			AnIdxTriangle tri(idx3, idx2, idx1);
			myTriangles.push_back(tri);
		}*/


	}

	delete knn;
}

/*void OpenCVWrapper::CaptureOpenGL(int w, int h, std::string filename)
{
	cv::Mat img(cv::Size(w, h), CV_8UC4, cv::Scalar(0, 0, 0));

	glReadBuffer(GL_FRONT);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glReadPixels(0, 0, w, h, GL_BGRA, GL_UNSIGNED_BYTE, (void*)img.data);

	cv::flip(img, img, 0);

	cv::imwrite(filename, img);
}*/
