# The following is specifically for CLI, exported to become a command line interface.
from .ZMPY3D_CLI_ZM                import ZMPY3D_CLI_ZM
from .ZMPY3D_CLI_SuperA2B          import ZMPY3D_CLI_SuperA2B
from .ZMPY3D_CLI_ShapeScore        import ZMPY3D_CLI_ShapeScore
from .ZMPY3D_CLI_BatchSuperA2B     import ZMPY3D_CLI_BatchSuperA2B
from .ZMPY3D_CLI_BatchZM           import ZMPY3D_CLI_BatchZM
from .ZMPY3D_CLI_BatchShapeScore   import ZMPY3D_CLI_BatchShapeScore


# The following renames and exports all libraries.
# 1-file-1-function (and they have the same name), good for future optimisation.
# use for Zernike moment
from .lib.calculate_ab_rotation_02              import calculate_ab_rotation_02                 as calculate_ab_rotation
from .lib.calculate_bbox_moment06               import calculate_bbox_moment06                  as calculate_bbox_moment
from .lib.calculate_bbox_moment_2_zm05          import calculate_bbox_moment_2_zm05             as calculate_bbox_moment_2_zm
from .lib.calculate_box_by_grid_width           import calculate_box_by_grid_width              as calculate_box_by_grid_width
from .lib.calculate_molecular_radius03          import calculate_molecular_radius03             as calculate_molecular_radius
from .lib.calculate_zm_by_ab_rotation01         import calculate_zm_by_ab_rotation01            as calculate_zm_by_ab_rotation
from .lib.eigen_root                            import eigen_root                               as eigen_root                         
from .lib.fill_voxel_by_weight_density04        import fill_voxel_by_weight_density04           as fill_voxel_by_weight_density
from .lib.get_3dzd_121_descriptor02             import get_3dzd_121_descriptor02                as get_3dzd_121_descriptor
from .lib.get_bbox_moment_xyz_sample01          import get_bbox_moment_xyz_sample01             as get_bbox_moment_xyz_sample
from .lib.get_global_parameter02                import get_global_parameter02                   as get_global_parameter
from .lib.get_mean_invariant03                  import get_mean_invariant03                     as get_mean_invariant
from .lib.get_residue_gaussian_density_cache02  import get_residue_gaussian_density_cache02     as get_residue_gaussian_density_cache
from .lib.get_residue_radius_map01              import get_residue_radius_map01                 as get_residue_radius_map
from .lib.get_residue_weight_map01              import get_residue_weight_map01                 as get_residue_weight_map

# used for shape score.
from .lib.get_total_residue_weight              import get_total_residue_weight                 as get_total_residue_weight
from .lib.get_descriptor_property               import get_descriptor_property                  as get_descriptor_property
from .lib.get_ca_distance_info                  import get_ca_distance_info                     as get_ca_distance_info

# used for superposition.
from .lib.calculate_ab_rotation_02_all          import calculate_ab_rotation_02_all             as calculate_ab_rotation_all
from .lib.get_transform_matrix_from_ab_list02   import get_transform_matrix_from_ab_list02      as get_transform_matrix_from_ab_list

# IO tools
from .lib.set_pdb_xyz_rot_m_01                  import set_pdb_xyz_rot_m_01                     as set_pdb_xyz_rot
from .lib.get_pdb_xyz_ca02                      import get_pdb_xyz_ca02                         as get_pdb_xyz_ca


# # used for EM, depends on MRCFILE, hence not imported
# from .lib.Voxel3D2MRCFile                   import Voxel3D2MRCFile                      as Voxel3D2MRCFile

# Add to __init__.py or a new config.py

def configure_for_scientific_computing(
    enable_x64: bool = True,
    platform: str = 'CPU',  # None = auto, 'cpu', 'gpu', 'tpu'
):
    """
    Configure JAX for scientific computing with ZMPY3D_JAX.
    
    Parameters
    ----------
    enable_x64 : bool, default=True
        Enable float64 precision. Critical for accurate Zernike moment calculations.
    platform : str, optional
        Force specific platform ('cpu', 'gpu', 'tpu'). None uses JAX default.
    
    Notes
    -----
    This function should be called ONCE at program startup, before any JAX operations.
    Float64 precision is strongly recommended for ZMPY3D_JAX to avoid numerical errors
    in iterative algorithms and accumulations.
    
    Examples
    --------
    >>> import ZMPY3D_JAX as z
    >>> z.configure_for_scientific_computing()  # Recommended
    >>> # Now use the library...
    """
    import jax
    
    if enable_x64:
        jax.config.update('jax_enable_x64', True)
        print("JAX configured for float64 precision")
    
    if platform is not None:
        jax.config.update('jax_platform_name', platform.lower())
        print(f"JAX configured for platform: {platform}")

