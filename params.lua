
---------- AnimationPak ----------
--- Title of the window
_window_title        = "1";
--- A directory where we have to save output files
_save_folder         = "C:\\Users\\azer\\OneDrive\\Images\\PhysicsPak_Snapshots_0" .. _window_title .. "\\";

_element_folder      = "C:\\Users\\azer\\OneDrive\\Images\\_animation_pak_data\\cat_rabbit\\";
_element_file_name   = "C:\\Users\\azer\\OneDrive\\Images\\_animation_pak_data\\bear.path"; -- NOT USED ANYMORE
_container_file_name = "C:\\Users\\azer\\OneDrive\\Images\\_animation_pak_data\\containers\\pp_heart_2.path";

--- artboard dimension (do not edit this)
--- the parameter below means the artboard size is 500x500
_upscaleFactor   = 500.0;                
_downscaleFactor = 1.0 / _upscaleFactor;

--- Time step for numerical integration (euler method)
_dt = 0.2;   --- do not set this higher than 0.1

--- random seed
_seed = -1; --- negative means random

--- Force parameters
_k_edge                = 1;	--- 0.5 edge force for filling elements
_k_z                   = 1;   --- preventing layers to stray away in z direction
_k_time_edge           = 0.001;
_k_neg_space_edge      = 0.01;	--- edge force for springs
_k_edge_small_factor   = 12;
_k_repulsion           = 3;	--- 10 repulsion force
_repulsion_soft_factor = 0.001;	--- soft factor for repulsion force
_k_overlap             = 0.1;	    --- overlap force
_k_boundary            = 0.1;	--- 0.1 boundary force
--_k_rotate              = 0;		--- 1
_k_dock                = 1.0;

--- capping the velocity
_velocity_cap   = 20; -- [Do not edit]

--- Grid for collision detection
--- size of a cell
_bin_square_size         = 20; -- 25
--- cell gap for detection, 
--- 1 means considering all cells that are 1 block away from the query (3x3)
--- 2 means considering all cells that are 2 block away from the query (5x5)
--_collission_block_radius = 1;   -- one means checking 3x3 
_grid_radius_1 = 2; -- exact
_grid_radius_2 = 3; -- approx (Barnes-Hut)
--_grid_radius_1_z = 1; -- NOT USED exact
--_grid_radius_2_z = 1; -- NOT USED approx (Barnes-Hut)

-- check A3DSquare.h and Mass.cpp
-- if error, use Debug mode to detect array out of bound
_max_exact_array_len  = 5000; 
_max_approx_array_len = 10000;

--- for growing
_growth_scale_iter     = 0.00075; -- 0.005
_element_initial_scale = 0.1; 
_element_max_scale     = 2.0;

-- num layer in the simulation, not the png layers
_num_layer = 50;

_interpolation_factor = 5; -- how many interpolation between two layers
_num_png_frame = 200;


---_show_time_springs = true;

--- density of random points inside the skin
--- if the density is higher, you get more triangles
_sampling_density               = 45;
--- uniform sampling on the skin
_boundary_sampling_factor   = 1.2;  -- [Do not edit]


-- for frame rendering


_num_element_density       = 200;
_num_element_pos_limit = 20;


