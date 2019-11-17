

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
	static std::string _static_element_folder;
	static std::string _animated_element_folder;
	//static std::string _element_file_name;
	static std::string _container_file_name;

	// because square
	static float _upscaleFactor;
	static float _downscaleFactor;

	/*
	_num_thread_cg = 12; -- collision grid
	_num_thread_springs = 4;
	*/
	//static int _num_thread_cg;
	//static int _num_thread_springs;
	//static int _num_thread_c_pt;
	//static int _num_thread_solve;
	static int _num_threads;

	static float _dt;

	static int _seed;

	//static float _k_edge;
	static float _k_edge_start;
	static float _k_edge_end;
	
	static float _k_z;
	static float _k_time_edge;
	static float _k_neg_space_edge;
	//static float _k_edge_small_factor;
	static float _k_repulsion;
	
	static float _k_overlap;
	static float _k_boundary;
	//static float _k_rotate;
	static float _k_dock;

	static float _k_aux_threshold;
	static float _k_aux_val_2_factor;
	static float _k_neg_space_threshold;
	static float _k_repulsion_soft_factor;

	static float _velocity_cap;

	//static float _self_intersection_threshold;

	static float _bin_square_size;
	//static int   _collission_block_radius;
	static int _grid_radius_1;
	static int _grid_radius_2;
	//static int _grid_radius_1_z;
	//static int _grid_radius_2_z;

	static int _max_cg_c_pts_len;
	static int _max_cg_c_pts_approx_len;

	static int _max_m_c_pts_len;
	static int _max_m_c_pts_approx_len;


	// temp
	static int _num_layer; // plus one

	//static bool _show_time_springs;

	static float _growth_scale_iter;
	static float _element_max_scale;
	static float _element_initial_scale;

	static float _growth_min_dist;

	// ttriangularization
	static float _skin_offset; // d_gap over 2
	static float _sampling_density;
	static float _boundary_sampling_factor;

	//static int _interpolation_factor;
	static int _num_png_frame;

	static int _num_element_density;
	static int _num_element_pos_limit;

	// viz
	static bool _show_mass_list;
	static bool _show_element_boundaries;
	static bool _show_exact_repulsion_forces;
	static bool _show_approx_repulsion_forces;
	static bool _show_collision_grid;
	static bool _show_collision_grid_object;
	static bool _show_surface_tri;

	static bool _show_growing_elements;

	static bool _show_dock_points;
	
	static bool _show_layer_springs;          // 0
	static bool _show_time_springs;           // 1
	static bool _show_aux_springs;            // 2
	static bool _show_negative_space_springs; // 3

	static bool _multithread_test;

	//static bool _show_c_pt_cg;
	//static bool _show_c_pt_approx_cg;

	static bool _show_force;
	static bool _show_overlap;
	static bool _show_closest_tri;

	static bool _show_container;

	static int _layer_slider_int;

};

#endif
