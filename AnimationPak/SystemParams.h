

#ifndef __System_Params__
#define __System_Params__

#include <string>

class SystemParams
{
public:
	SystemParams();
	~SystemParams();

	static void LoadParameters();

public:
	static std::string _lua_file;
	static std::string _window_title;

	// because square
	static float _upscaleFactor;
	static float _downscaleFactor;

	static float _dt;

	static int _seed;

	static float _k_edge;
	static float _k_neg_space_edge;
	static float _k_edge_small_factor;
	static float _k_repulsion;
	static float _repulsion_soft_factor;
	static float _k_overlap;
	static float _k_boundary;
	static float _k_rotate;
	static float _k_dock;

	static float _velocity_cap;
};

#endif
