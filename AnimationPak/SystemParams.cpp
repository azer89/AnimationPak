

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
	//std::string lua_file = "..\\params.lua";
	LuaScript script(_lua_file);

	SystemParams::_upscaleFactor = script.get<float>("_upscaleFactor");
	SystemParams::_downscaleFactor = script.get<float>("_downscaleFactor");

	//std::cout << "upscale factor = " << SystemParams::_upscaleFactor << "\n";
	//std::cout << "downscale factor = " << SystemParams::_downscaleFactor << "\n";

	SystemParams::_seed = script.get<int>("_seed");

	SystemParams::_k_edge = script.get<float>("_k_edge");
	SystemParams::_k_neg_space_edge = script.get<float>("_k_neg_space_edge");
	SystemParams::_k_edge_small_factor = script.get<float>("_k_edge_small_factor");
	SystemParams::_k_repulsion = script.get<float>("_k_repulsion");
	SystemParams::_repulsion_soft_factor = script.get<float>("_repulsion_soft_factor");
	SystemParams::_k_overlap = script.get<float>("_k_overlap");
	SystemParams::_k_boundary = script.get<float>("_k_boundary");
	SystemParams::_k_rotate = script.get<float>("_k_rotate");
	SystemParams::_k_dock = script.get<float>("_k_dock");

	SystemParams::_bin_square_size = script.get<float>("_bin_square_size");
	SystemParams::_collission_block_radius = script.get<int>("_collission_block_radius");

	SystemParams::_velocity_cap = script.get<float>("_velocity_cap");

	// temp
	//SystemParams::_cube_length = 500.0f;
	SystemParams::_num_layer = 6; // plus one


	//std::cout << SystemParams::_upscaleFactor << "\n";
	//std::cout << SystemParams::_downscaleFactor << "\n";
	//std::cout << SystemParams::_seed << "\n";

}

float SystemParams::_upscaleFactor = 0.0f;
float SystemParams::_downscaleFactor = 0.0f;

float SystemParams::_dt = 0.0f;

int SystemParams::_seed = 0;

float SystemParams::_k_edge = 0.0f;
float SystemParams::_k_neg_space_edge = 0.0f;
float SystemParams::_k_edge_small_factor = 0.0f;
float SystemParams::_k_repulsion = 0.0f;
float SystemParams::_repulsion_soft_factor = 0.0f;
float SystemParams::_k_overlap = 0.0f;
float SystemParams::_k_boundary = 0.0f;
float SystemParams::_k_rotate = 0.0f;
float SystemParams::_k_dock = 0.0f;

float SystemParams::_velocity_cap = 0.0f;

float SystemParams::_bin_square_size = 0.0f;
int   SystemParams::_collission_block_radius = 0;

// temp
int SystemParams::_num_layer = 0.0f; // plus one
