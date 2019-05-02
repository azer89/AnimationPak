
---------- AnimationPak ----------
--- Title of the window
_window_title = "1" 
--- A directory where we have to save output files
_save_folder  = "C:\\Users\\azer\\OneDrive\\Images\\PhysicsPak_Snapshots_0" .. _window_title .. "\\"

--- artboard dimension (do not edit this)
--- the parameter below means the artboard size is 500x500
_upscaleFactor   = 500.0;                
_downscaleFactor = 1.0 / _upscaleFactor;

--- Time step for numerical integration (euler method)
_dt = 0.1;   --- do not set this higher than 0.1

--- random seed
_seed = -1; --- negative means random

--- Force parameters
_k_edge                = 1.0;	--- 0.5 edge force for filling elements
_k_time_edge           = 0.01;
_k_neg_space_edge      = 10;	--- edge force for springs
_k_edge_small_factor   = 12;
_k_repulsion           = 100;	--- 200 repulsion force
_repulsion_soft_factor = 1.0;	--- soft factor for repulsion force
_k_overlap             = 100;	--- overlap force
_k_boundary            = 0.3;	--- 0.1 boundary force
_k_rotate              = 1;		--- 1
_k_dock                = 1.0;

--- capping the velocity
_velocity_cap   = 10; -- [Do not edit]

--- Grid for collision detection
--- size of a cell
_bin_square_size         = 250;
--- cell gap for detection, 
--- 1 means considering all cells that are 1 block away from the query (3x3)
--- 2 means considering all cells that are 2 block away from the query (5x5)
_collission_block_radius = 1;  

--- for growing
_growth_scale_iter     = 0.0001; -- 0.005
_element_max_scale = 1.29;

_num_layer = 101;


_show_time_springs = true;

--- density of random points inside the skin
--- if the density is higher, you get more triangles
_sampling_density               = 200;
--- uniform sampling on the skin
_boundary_sampling_factor   = 1.2;  -- [Do not edit]


