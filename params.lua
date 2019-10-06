
---------- AnimationPak ----------
--- Title of the window
_window_title        = "1";
--- A directory where we have to save output files
_save_folder         = "C:\\Users\\azer\\OneDrive\\Images\\PhysicsPak_Snapshots_0" .. _window_title .. "\\";

_element_folder      = "C:\\Users\\azer\\OneDrive\\Images\\_animation_pak_data\\stars\\";
_element_file_name   = "C:\\Users\\azer\\OneDrive\\Images\\_animation_pak_data\\bear.path"; -- NOT USED ANYMORE
_container_file_name = "C:\\Users\\azer\\OneDrive\\Images\\_animation_pak_data\\containers\\circle.path";

--- artboard dimension (do not edit this)
--- the parameter below means the artboard size is 500x500
_upscaleFactor   = 500.0;                
_downscaleFactor = 1.0 / _upscaleFactor;

--- Time step for numerical integration (euler method)
_dt = 0.1;   --- do not set this higher than 0.1

--- random seed
_seed = -1; --- negative means random


---
_num_thread_cg      = 12; -- collision grid
_num_thread_springs = 10;
_num_thread_c_pt    = 20; -- closest point
_num_thread_solve   = 20;

--- Force parameters
_k_edge_start          = 8.0;
_k_edge_end            = 2.0;
--_k_edge              = 4.0;	--- 0.5 edge force for filling elements
_k_z                   = 1.0;   --- preventing layers to stray away in z direction
_k_time_edge           = 0.01;
_k_neg_space_edge      = 0.1;	--- edge force for springs
--_k_edge_small_factor = 12;
_k_repulsion           = 2.0;	--- 10 repulsion force
_repulsion_soft_factor = 0.001;	--- soft factor for repulsion force
_k_overlap             = 1;	    --- overlap force
_k_boundary            = 0.5;	--- 0.1 boundary force
--_k_rotate              = 0;		--- 1
_k_dock                = 1.0;

-- Activating negative space springs
_self_intersection_threshold = 2.0;

--- capping the velocity
_velocity_cap   = 20; -- [Do not edit]

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
_max_exact_array_len  = 5000; 
_max_approx_array_len = 10000;

--- for growing
_growth_scale_iter     = 0.005; -- 0.005
_element_initial_scale = 0.1; 
_element_max_scale     = 3.1;

-- num layer in the simulation, not the png layers
_num_layer = 100;

--_interpolation_factor = 5; -- how many interpolation between two layers

_num_png_frame = 500;

---_show_time_springs = true;
_skin_offset              = 20;  

-- For triangulation
--- density of random points inside the skin
--- if the density is higher, you get more triangles
_sampling_density         = 100; -- For triangulation

--- uniform sampling on the skin
_boundary_sampling_factor = 1.2;  -- lower means denser


-- random point density, NOT triangulation !!!
_num_element_density   = 200;

-- number of elements
_num_element_pos_limit = 25;


