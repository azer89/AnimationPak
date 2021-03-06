

/* ---------- ShapeRadiusMatching V2  ---------- */

#include "SystemParams.h"

#include <sstream>

#include "LuaScript.h"


std::string SystemParams::_lua_file = "..\\params.lua";

SystemParams::SystemParams()
{
}

SystemParams::~SystemParams()
{
}

void SystemParams::LoadParameters()
{
	

	LuaScript script(_lua_file);

	SystemParams::_window_title            = script.get<std::string>("_window_title");
	SystemParams::_save_folder             = script.get<std::string>("_save_folder");
	SystemParams::_animated_element_folder = script.get<std::string>("_animated_element_folder");
	SystemParams::_static_element_folder   = script.get<std::string>("_static_element_folder");
	//SystemParams::_element_file_name     = script.get<std::string>("_element_file_name");
	SystemParams::_container_file_name     = script.get<std::string>("_container_file_name");
	SystemParams::_scene_file_name = script.get<std::string>("_scene_file_name");

	SystemParams::_upscaleFactor = script.get<float>("_upscaleFactor");
	SystemParams::_downscaleFactor = script.get<float>("_downscaleFactor");

	SystemParams::_dt = script.get<float>("_dt");
	SystemParams::_seed = script.get<int>("_seed");

	SystemParams::_num_threads = script.get<int>("_num_threads");
	//SystemParams::_num_thread_cg = script.get<int>("_num_thread_cg");
	//SystemParams::_num_thread_springs = script.get<int>("_num_thread_springs");
	//SystemParams::_num_thread_c_pt = script.get<int>("_num_thread_c_pt");
	//SystemParams::_num_thread_solve = script.get<int>("_num_thread_solve");

	SystemParams::_k_connector = script.get<float>("_k_connector");
	//SystemParams::_k_edge = script.get<float>("_k_edge");
	SystemParams::_k_edge_start = script.get<float>("_k_edge_start");
	SystemParams::_k_edge_end = script.get<float>("_k_edge_end");


	SystemParams::_k_z = script.get<float>("_k_z");
	SystemParams::_k_time_edge = script.get<float>("_k_time_edge");
	SystemParams::_k_neg_space_edge = script.get<float>("_k_neg_space_edge");
	//SystemParams::_k_edge_small_factor = script.get<float>("_k_edge_small_factor");
	SystemParams::_k_repulsion = script.get<float>("_k_repulsion");
	//SystemParams::_repulsion_soft_factor = script.get<float>("_repulsion_soft_factor");
	SystemParams::_k_overlap = script.get<float>("_k_overlap");
	SystemParams::_k_boundary = script.get<float>("_k_boundary");
	SystemParams::_k_rotate = script.get<float>("_k_rotate");
	SystemParams::_k_dock = script.get<float>("_k_dock");

	SystemParams::_k_aux_val_2_factor = script.get<float>("_k_aux_val_2_factor");
	SystemParams::_k_aux_threshold = script.get<float>("_k_aux_threshold");
	SystemParams::_k_neg_space_threshold = script.get<float>("_k_neg_space_threshold");
	SystemParams::_k_repulsion_soft_factor = script.get<float>("_k_repulsion_soft_factor");

	SystemParams::_bin_square_size = script.get<float>("_bin_square_size");
	SystemParams::_grid_radius_1 = script.get<int>("_grid_radius_1");
	SystemParams::_grid_radius_2 = script.get<int>("_grid_radius_2");
	//SystemParams::_grid_radius_1_z = script.get<int>("_grid_radius_1_z");
	//SystemParams::_grid_radius_2_z = script.get<int>("_grid_radius_2_z");
	SystemParams::_max_cg_c_pts_len = script.get<int>("_max_cg_c_pts_len");
	SystemParams::_max_cg_c_pts_approx_len = script.get<int>("_max_cg_c_pts_approx_len");

	SystemParams::_max_m_c_pts_len = script.get<int>("_max_m_c_pts_len");
	SystemParams::_max_m_c_pts_approx_len = script.get<int>("_max_m_c_pts_approx_len");
	//SystemParams::_collission_block_radius = script.get<int>("_collission_block_radius");

	SystemParams::_velocity_cap = script.get<float>("_velocity_cap");

	//SystemParams::_self_intersection_threshold = script.get<float>("_self_intersection_threshold");

	// temp
	//SystemParams::_cube_length = 500.0f;
	SystemParams::_num_layer = script.get<int>("_num_layer"); // plus one
	//SystemParams::_show_time_springs = script.get<bool>("_show_time_springs");
	SystemParams::_growth_scale_iter = script.get<float>("_growth_scale_iter");
	SystemParams::_element_max_scale = script.get<float>("_element_max_scale");
	SystemParams::_element_initial_scale = script.get<float>("_element_initial_scale");

	SystemParams::_growth_min_dist = script.get<float>("_growth_min_dist");

	SystemParams::_sampling_density = script.get<float>("_sampling_density");
	SystemParams::_boundary_sampling_factor = script.get<float>("_boundary_sampling_factor");

	//SystemParams::_interpolation_factor = script.get<int>("_interpolation_factor");
	SystemParams::_num_png_frame = script.get<int>("_num_png_frame");

	SystemParams::_skin_offset = script.get<float>("_skin_offset");
	SystemParams::_num_element_density = script.get<int>("_num_element_density");
	SystemParams::_num_element_pos_limit = script.get<int>("_num_element_pos_limit");

	//std::cout << SystemParams::_upscaleFactor << "\n";
	//std::cout << SystemParams::_downscaleFactor << "\n";
	//std::cout << SystemParams::_seed << "\n";
	//std::cout << SystemParams::_window_title << "\n";
	//std::cout << SystemParams::_save_folder << "\n";

	SystemParams::_num_layer_growing_threshold = script.get<int>("_num_layer_growing_threshold");

	std::stringstream ss;
	ss << "copy " << _lua_file << " " << SystemParams::_save_folder << "params.lua";
	std::cout << ss.str() << "\n";
	std::system(ss.str().c_str());

}

std::string SystemParams::_window_title        = "";
std::string SystemParams::_animated_element_folder = "";
std::string SystemParams::_static_element_folder      = "";
std::string SystemParams::_save_folder         = "";
//std::string SystemParams::_element_file_name   = "";
std::string SystemParams::_container_file_name = "";
std::string SystemParams::_scene_file_name = "";

float SystemParams::_upscaleFactor = 0.0f;
float SystemParams::_downscaleFactor = 0.0f;

float SystemParams::_dt = 0.0f;

int SystemParams::_seed = 0;

int SystemParams::_num_threads = 0;
//int SystemParams::_num_thread_cg = 0;
//int SystemParams::_num_thread_springs = 0;
//int SystemParams::_num_thread_c_pt = 0;
//int SystemParams::_num_thread_solve = 0;

//float SystemParams::_k_edge = 0.0f;
float SystemParams::_k_connector = 0.0f;
float SystemParams::_k_edge_start = 0.0f;
float SystemParams::_k_edge_end = 0.0f;

float SystemParams::_k_z = 0.0f;
float SystemParams::_k_time_edge = 0.0f;
float SystemParams::_k_neg_space_edge = 0.0f;
//float SystemParams::_k_edge_small_factor = 0.0f;
float SystemParams::_k_repulsion = 0.0f;

float SystemParams::_k_overlap = 0.0f;
float SystemParams::_k_boundary = 0.0f;
float SystemParams::_k_rotate = 0.0f;
float SystemParams::_k_dock = 0.0f;

float SystemParams::_k_aux_threshold = 0.0f;
float SystemParams::_k_aux_val_2_factor = 0.0f;
float SystemParams::_k_neg_space_threshold = 0.0f;
float SystemParams::_k_repulsion_soft_factor = 0.0f;

float SystemParams::_velocity_cap = 0.0f;

//float SystemParams::_self_intersection_threshold = 0.0f;

float SystemParams::_bin_square_size = 0.0f;
int SystemParams::_grid_radius_1 = 0;
int SystemParams::_grid_radius_2 = 0;
//int SystemParams::_grid_radius_1_z = 0;
//int SystemParams::_grid_radius_2_z = 0;
int SystemParams::_max_cg_c_pts_len = 0;
int SystemParams::_max_cg_c_pts_approx_len = 0;

int SystemParams::_max_m_c_pts_len = 0;
int SystemParams::_max_m_c_pts_approx_len = 0;
//int   SystemParams::_collission_block_radius = 0;



// temp
int SystemParams::_num_layer = 0.0f; // plus one

//bool  SystemParams::_show_time_springs = false;

float SystemParams::_growth_scale_iter = 0.0f;
float SystemParams::_element_max_scale = 0.0f;
float SystemParams::_element_initial_scale = 0.0f;
float SystemParams::_growth_min_dist = 0.0f;

float SystemParams::_skin_offset = 0.0f;
float SystemParams::_sampling_density = 0.0f;
float SystemParams::_boundary_sampling_factor = 0.0f;

//int SystemParams::_interpolation_factor = 0;
int SystemParams::_num_png_frame = 0;

int SystemParams::_num_element_density = 0;
int SystemParams::_num_element_pos_limit = 0;

// viz
bool SystemParams::_show_mass_list = false;
bool SystemParams::_show_element_boundaries = true;
bool SystemParams::_show_exact_repulsion_forces = false;
bool SystemParams::_show_approx_repulsion_forces = false;
bool SystemParams::_show_collision_grid = false;
bool SystemParams::_show_collision_grid_object = false;
bool SystemParams::_show_surface_tri = false;
bool SystemParams::_show_arts = false;
bool SystemParams::_show_centers = false;

bool SystemParams::_show_growing_elements = false;

bool SystemParams::_show_dock_points = false;

bool SystemParams::_multithread_test = false;

bool SystemParams::_show_layer_springs = false;          // 0
bool SystemParams::_show_time_springs = false;           // 1
bool SystemParams::_show_aux_springs = false;            // 2
bool SystemParams::_show_negative_space_springs = false; // 3

//bool SystemParams::_show_c_pt_cg        = false;
//bool SystemParams::_show_c_pt_approx_cg = false;

bool SystemParams::_show_force       = false;
bool SystemParams::_show_overlap     = false;
bool SystemParams::_show_closest_tri = false;

bool SystemParams::_show_container   = true;

int SystemParams::_layer_slider_int  = -1;

int SystemParams::_num_layer_growing_threshold = 0;