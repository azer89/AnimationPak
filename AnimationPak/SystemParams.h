

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
	static std::string _save_folder;
	static std::string _element_file_name;
	static std::string _container_file_name;

	// because square
	static float _upscaleFactor;
	static float _downscaleFactor;

	static float _dt;

	static int _seed;

	static float _k_edge;
	static float _k_z;
	static float _k_time_edge;
	static float _k_neg_space_edge;
	static float _k_edge_small_factor;
	static float _k_repulsion;
	static float _repulsion_soft_factor;
	static float _k_overlap;
	static float _k_boundary;
	static float _k_rotate;
	static float _k_dock;

	static float _velocity_cap;

	static float _bin_square_size;
	//static int   _collission_block_radius;
	static int _grid_radius_1;
	static int _grid_radius_2;

	static int _max_exact_array_len;
	static int _max_approx_array_len;


	// temp
	static int _num_layer; // plus one

	//static bool _show_time_springs;

	static float _growth_scale_iter;
	static float _element_max_scale;
	static float _element_initial_scale;

	// tetrathedralization
	static float _sampling_density;
	static float _boundary_sampling_factor;

	static int _interpolation_factor;

	static int _num_element_density;
	static int _num_element_pos_limit;

	// viz
	static bool _show_collision_grid;
	static bool _show_time_springs;
};

#endif
