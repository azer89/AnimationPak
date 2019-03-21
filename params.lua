
---------- AnimationPak ----------
--- Title of the window
_window_title = "1" 
--- A directory where we have to save output files
--_save_folder  = "C:\\Users\\azer\\OneDrive\\Images\\PhysicsPak_Snapshots_0" .. _window_title .. "\\"

--- artboard dimension (do not edit this)
--- the parameter below means the artboard size is 500x500
_upscaleFactor   = 500.0;                
_downscaleFactor = 1.0 / _upscaleFactor;

--- Time step for numerical integration (euler method)
_dt = 0.05;   --- do not set this higher than 0.1

--- random seed
_seed = -1; --- negative means random

--- Force parameters
_k_edge                = 40;	--- edge force for filling elements
_k_neg_space_edge      = 10;	--- edge force for springs
_k_edge_small_factor   = 12;
_k_repulsion           = 60;	--- repulsion force
_repulsion_soft_factor = 1.0;	--- soft factor for repulsion force
_k_overlap             = 5;	--- overlap force
_k_boundary            = 5;	--- boundary force
_k_rotate              = 1;		--- 1
_k_dock                = 100;

--- capping the velocity
_velocity_cap   = 5; -- [Do not edit]