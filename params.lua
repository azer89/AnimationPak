
---------- AnimationPak ----------
--- Title of the window
_window_title        = "1";
--- A directory where we have to save output files
_save_folder         = "C:\\Users\\azer\\OneDrive\\Images\\PhysicsPak_Snapshots_0" .. _window_title .. "\\";
_animated_element_folder= "C:\\Users\\azer\\OneDrive\\Images\\_animation_pak_data\\chase_bird_ani\\"; 
_static_element_folder      = "C:\\Users\\azer\\OneDrive\\Images\\_animation_pak_data\\stars\\";
--_element_file_name   = "C:\\Users\\azer\\OneDrive\\Images\\_animation_pak_data\\bear.path"; -- NOT USED ANYMORE
_container_file_name = "C:\\Users\\azer\\OneDrive\\Images\\_animation_pak_data\\containers\\donut_2.path";
_scene_file_name = "C:\\Users\\azer\\OneDrive\\Images\\_animation_pak_data\\containers\\donut_2.scene";

--- artboard dimension (do not edit this)
--- the parameter below means the artboard size is 500x500
_upscaleFactor   = 500.0;                
_downscaleFactor = 1.0 / _upscaleFactor;

--- Time step for numerical integration (euler method)
_dt = 0.05; --- 0.05 for good result? (Do not set this higher than 0.1)

--- random seed
_seed = -1; --- negative means random


---
_num_threads = 12;
--_num_thread_cg      = 12; -- collision grid
--_num_thread_springs = 4;
--_num_thread_c_pt    = 12; -- closest point
--_num_thread_solve   = 12;

--- Force parameters
_k_connector           = 10;
_k_edge_start          = 20;   -- 5 edge force for filling elements
_k_edge_end            = 4;     --- 5 edge force for filling elements
--_k_edge              = 40;	--- 0.5 edge force for filling elements
_k_z                   = 1;   --- preventing layers to stray away in z direction
_k_time_edge           = 0.002;
_k_neg_space_edge      = 0.1;	--- 0.01 edge force for springs
--_k_edge_small_factor = 12;
_k_repulsion           = 10.0;	--- 10 repulsion force
_k_overlap             = 5;	    --- overlap force
_k_boundary            = 5;	--- 0.1 boundary force
_k_rotate              = 0.0;		--- 1
_k_dock                = 0.1;


_k_aux_threshold = 0.7;
_k_aux_val_2_factor = 8;
_k_neg_space_threshold = 2;
_k_repulsion_soft_factor = 1;	--- soft factor for repulsion force

-- Activating negative space springs
--_self_intersection_threshold = 2.0;

--- capping the velocity
_velocity_cap   = 10; -- [Do not edit]

--- Grid for collision detection
--- size of a cell
_bin_square_size         = 20; -- 25
--- cell gap for detection, 
--- 1 means considering all cells that are 1 block away from the query (3x3)
--- 2 means considering all cells that are 2 block away from the query (5x5)
--_collission_block_radius = 1;   -- one means checking 3x3 
_grid_radius_1 = 1; -- exact
_grid_radius_2 = 2; -- approx (Barnes-Hut)
--_grid_radius_1_z = 1; -- NOT USED exact
--_grid_radius_2_z = 1; -- NOT USED approx (Barnes-Hut)

-- check A3DSquare.h and Mass.cpp
-- if error, use Debug mode to detect array out of bound
_max_cg_c_pts_len  = 7000; 
_max_cg_c_pts_approx_len = 10000;

_max_m_c_pts_len  = 2000; 
_max_m_c_pts_approx_len = 2000;

--- for growing
_growth_scale_iter     = 0.01; -- 0.01
_element_initial_scale = 0.09; 
_element_max_scale     = 15.0;  -- 8.25

--- epsilon for halting the growth
_growth_min_dist       = 3; 

-- num layer in the simulation, not the png layers
_num_layer = 100;

--_interpolation_factor = 5; -- how many interpolation between two layers

_num_png_frame = 500;

---_show_time_springs = true;
_skin_offset              = 10;  

-- For triangulation
--- density of random points inside the skin
--- if the density is higher, you get more triangles
_sampling_density         = 150; -- For triangulation

--- uniform sampling on the skin
_boundary_sampling_factor = 1.4;  -- lower means denser

-- TODO distance to boundary so that elements do not protrude outside

-- CONSIDER EDITING BOTH !!!
-- number of elements
_num_element_pos_limit = 1;
-- random point density, NOT triangulation !!!
_num_element_density   = 100;

-- stopping criteria
_num_layer_growing_threshold = 50; --100

