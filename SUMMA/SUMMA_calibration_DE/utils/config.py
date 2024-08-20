# config.py

from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime
from pathlib import Path
from mpi4py import MPI # type: ignore
import numpy as np # type: ignore
from utils.calibration_utils import read_param_bounds # type: ignore
from pathlib import Path
from datetime import datetime

def read_from_control(file, setting):
    """Extract a given setting from the control file."""
    with open(file) as contents:
        for line in contents:
            if setting in line and not line.startswith('#'):
                return line.split('|', 1)[1].split('#', 1)[0].strip()
    return None

def is_default_path(control_folder, control_file, path, suffix):
    if path == 'default':
        return(make_default_path(control_folder, control_file, suffix))
    else:
        return Path(path)

def make_default_path(control_folder, control_file, suffix):
    """Specify a default path based on the control file settings."""
    root_path = Path(read_from_control(control_folder/control_file, 'root_path'))
    domain_name = read_from_control(control_folder/control_file, 'domain_name')
    return root_path / f'domain_{domain_name}' / suffix

def parse_time_period(period_str):
    """Parse time period string into datetime objects."""
    start, end = [datetime.strptime(date.strip(), '%Y-%m-%d') for date in period_str.split(',')]
    return start, end

def get_config_path(control_folder, control_file, setting, default_suffix=None, is_folder=False):
    """
    Get a configuration path for a file or folder, using a default if specified.
    
    Args:
    control_folder (Path): Path to the folder containing the control file
    control_file (str): Name of the control file
    setting (str): The setting to read from the control file
    default_suffix (str, optional): The suffix to append to the default path if 'default' is specified
    is_folder (bool): Whether the path is for a folder (True) or file (False)
    
    Returns:
    Path: The configuration path
    """
    path = read_from_control(control_folder/control_file, setting)
    if path == 'default':
        if is_folder:
            root_path = Path(read_from_control(control_folder/control_file, 'root_code_path'))
            return root_path / default_suffix
        else:
            return path
    return Path(path)

@dataclass
class Config:
    root_path: str
    domain_name: str
    experiment_id: str
    params_to_calibrate: List[str]
    basin_params_to_calibrate: List[str]
    obs_file_path: str
    sim_reach_ID: str
    filemanager_name: str
    optimization_metrics: List[str]
    pop_size: int
    mizu_control_file: str
    moo_num_iter: int
    nsga2_n_gen: int
    nsga2_n_obj: int
    optimization_metric: str
    algorithm: str
    num_iter: int
    calib_period: Tuple[datetime, datetime]
    eval_period: Tuple[datetime, datetime]
    local_bounds_dict: dict
    basin_bounds_dict: dict
    local_bounds: List[Tuple[float, float]]
    basin_bounds: List[Tuple[float, float]]
    all_bounds: List[Tuple[float, float]]
    all_params: List[str]
    poplsize: int
    swrmsize: int
    ngsize: int
    dds_r: float
    diagnostic_frequency: int
    snow_processed_path: Path
    snow_processed_name: str
    snow_station_shapefile_path: Path
    snow_station_shapefile_name: str
    MODIS_ndsi_threshold: int
    catchment_shp_name: str
    ostrich_path: Path
    ostrich_exe: str
    ostrich_algorithm: str
    ostrich_metric: str
    ostrich_swarm_size: int
    ostrich_output_file: str
    basin_parameters_file:str
    basin_parameters_file_name: str
    local_parameters_file_name: str
    trial_param_file: str
    settings_summa_attributes: str
    settings_summa_output: str
    settings_summa_filemanager: str
    exe_name_summa: str
    output_path: str
    output_prefix: str
    root_code_path: Path
    exe_name_mizuroute: str
    use_mpi: bool
    sce_budget: int
    sce_loop_stagnation: int
    sce_pct_change: float
    sce_pop_conv: float
    sce_num_complexes: int
    sce_points_per_complex: int
    sce_points_per_subcomplex: int
    sce_num_evolution_steps: int
    sce_min_complexes: int
    sce_use_initial_point: str
    pso_swarm_size: int
    pso_num_generations: int
    pso_constriction_factor: float
    pso_cognitive_param: float
    pso_social_param: float
    pso_inertia_weight: float
    pso_inertia_reduction_rate: float
    pso_init_population_method: str
    pso_convergence_val: float
    num_ranks: int 
    

def initialize_config(rank: int, comm: MPI.Comm) -> Config:
    control_folder = Path('/Users/darrieythorsson/compHydro/code/CWARHM/0_control_files/')
    control_file = 'control_active.txt'

    if comm is None:
        # Create a dummy comm object for non-MPI usage
        class DummyComm:
            def bcast(self, obj, root=0):
                return obj
            def Get_rank(self):
                return 0
        comm = DummyComm()
        rank = 0

    if rank == 0:
        root_path = read_from_control(control_folder/control_file, 'root_path')
        domain_name = read_from_control(control_folder/control_file, 'domain_name')
        experiment_id = read_from_control(control_folder/control_file, 'experiment_id')
        params_to_calibrate = read_from_control(control_folder/control_file, 'params_to_calibrate').split(',')
        basin_params_to_calibrate = read_from_control(control_folder/control_file, 'basin_params_to_calibrate').split(',')
        obs_file_path = read_from_control(control_folder/control_file, 'obs_file_path')
        sim_reach_ID = read_from_control(control_folder/control_file, 'sim_reach_ID')
        filemanager_name = read_from_control(control_folder/control_file, 'settings_summa_filemanager')
        mizu_control_file = read_from_control(control_folder/control_file, 'settings_mizu_control_file')
        optimization_metric = read_from_control(control_folder/control_file, 'optimization_metric')
        algorithm = read_from_control(control_folder/control_file, 'Optimisation_algorithm')
        num_iter = int(read_from_control(control_folder/control_file, 'num_iter'))
        calib_period_str = read_from_control(control_folder/control_file, 'calibration_period')
        eval_period_str = read_from_control(control_folder/control_file, 'evaluation_period')
        calib_period = parse_time_period(calib_period_str)
        eval_period = parse_time_period(eval_period_str) 
        optimization_metrics = read_from_control(control_folder/control_file, 'moo_optimization_metrics').split(',')
        pop_size = int(read_from_control(control_folder/control_file, 'moo_pop_size'))
        moo_num_iter = int(read_from_control(control_folder/control_file, 'moo_num_iter'))
        nsga2_n_gen = int(read_from_control(control_folder/control_file, 'nsga2_n_gen'))
        nsga2_n_obj = int(read_from_control(control_folder/control_file, 'nsga2_n_obj'))
        local_parameters_file = make_default_path(control_folder, control_file, 'settings/SUMMA/localParamInfo.txt')
        basin_parameters_file = make_default_path(control_folder, control_file, 'settings/SUMMA/basinParamInfo.txt')
        basin_parameters_file_name = read_from_control(control_folder/control_file, 'basin_parameters_file_name')
        local_parameters_file_name = read_from_control(control_folder/control_file, 'local_parameters_file_name')
        local_bounds_dict = read_param_bounds(local_parameters_file, params_to_calibrate)
        basin_bounds_dict = read_param_bounds(basin_parameters_file, basin_params_to_calibrate)
        trial_param_file = read_from_control(control_folder/control_file, 'trial_param_file')

        local_bounds = [local_bounds_dict[param] for param in params_to_calibrate]
        basin_bounds = [basin_bounds_dict[param] for param in basin_params_to_calibrate]
        all_bounds = local_bounds + basin_bounds
        all_params = params_to_calibrate + basin_params_to_calibrate
        snow_processed_path = make_default_path(control_folder, control_file, 'observations/snow/preprocessed/')
        snow_processed_name = read_from_control(control_folder/control_file, 'snow_processed_name')
        snow_station_shapefile_path = make_default_path(control_folder, control_file, 'shapefiles/observations/')
        snow_station_shapefile_name = read_from_control(control_folder/control_file, 'snow_station_shapefile_name')

        poplsize = int(read_from_control(control_folder/control_file, 'poplsize'))
        swrmsize = int(read_from_control(control_folder/control_file, 'swrmsize'))
        ngsize = int(read_from_control(control_folder/control_file, 'ngsize'))
        dds_r = float(read_from_control(control_folder/control_file, 'dds_r'))
        diagnostic_frequency = int(read_from_control(control_folder/control_file, 'diagnostic_frequency'))
        MODIS_ndsi_threshold = int(read_from_control(control_folder/control_file, 'modis_ndsi_threshold'))
        catchment_shp_name = read_from_control(control_folder/control_file, 'catchment_shp_name')
        ostrich_path = Path(read_from_control(control_folder/control_file, 'ostrich_path'))
        ostrich_exe = read_from_control(control_folder/control_file, 'ostrich_exe')
        use_mpi = ostrich_exe.lower().endswith('mpi')
        ostrich_algorithm = read_from_control(control_folder/control_file, 'ostrich_algorithm')
        ostrich_metric = read_from_control(control_folder/control_file, 'ostrich_metric')
        ostrich_swarm_size = int(read_from_control(control_folder/control_file, 'ostrich_swarm_size'))
        ostrich_output_file = read_from_control(control_folder/control_file, 'ostrich_output_file')
        settings_summa_attributes = read_from_control(control_folder/control_file, 'settings_summa_attributes')
        settings_summa_output = read_from_control(control_folder/control_file, 'settings_summa_output')
        settings_summa_filemanager = read_from_control(control_folder/control_file, 'settings_summa_filemanager')
        exe_name_summa = read_from_control(control_folder/control_file, 'exe_name_summa')
        output_path = read_from_control(control_folder/control_file, 'output_path')
        output_prefix = read_from_control(control_folder/control_file, 'output_prefix')
        root_code_path = Path(read_from_control(control_folder/control_file, 'root_code_path'))
        exe_name_mizuroute = read_from_control(control_folder/control_file, 'exe_name_mizuroute')
        sce_budget = int(read_from_control(control_folder/control_file, 'sce_budget'))
        sce_loop_stagnation = int(read_from_control(control_folder/control_file, 'sce_loop_stagnation'))
        sce_pct_change = float(read_from_control(control_folder/control_file, 'sce_pct_change'))
        sce_pop_conv = float(read_from_control(control_folder/control_file, 'sce_pop_conv'))
        sce_num_complexes = int(read_from_control(control_folder/control_file, 'sce_num_complexes'))
        sce_points_per_complex = int(read_from_control(control_folder/control_file, 'sce_points_per_complex'))
        sce_points_per_subcomplex = int(read_from_control(control_folder/control_file, 'sce_points_per_subcomplex'))
        sce_num_evolution_steps = int(read_from_control(control_folder/control_file, 'sce_num_evolution_steps'))
        sce_min_complexes = int(read_from_control(control_folder/control_file, 'sce_min_complexes'))
        sce_use_initial_point = read_from_control(control_folder/control_file, 'sce_use_initial_point')
        pso_swarm_size = int(read_from_control(control_folder/control_file, 'pso_swarm_size'))
        pso_num_generations = int(read_from_control(control_folder/control_file, 'pso_num_generations'))
        pso_constriction_factor = float(read_from_control(control_folder/control_file, 'pso_constriction_factor'))
        pso_cognitive_param = float(read_from_control(control_folder/control_file, 'pso_cognitive_param'))
        pso_social_param = float(read_from_control(control_folder/control_file, 'pso_social_param'))
        pso_inertia_weight = float(read_from_control(control_folder/control_file, 'pso_inertia_weight'))
        pso_inertia_reduction_rate = float(read_from_control(control_folder/control_file, 'pso_inertia_reduction_rate'))
        pso_init_population_method = read_from_control(control_folder/control_file, 'pso_init_population_method')
        pso_convergence_val = float(read_from_control(control_folder/control_file, 'pso_convergence_val'))
        num_ranks = int(read_from_control(control_folder/control_file, 'num_ranks')) 

    else:
        ostrich_algorithm = None
        ostrich_metric = None
        ostrich_swarm_size = None
        ostrich_output_file = None
        root_path = None
        domain_name = None
        experiment_id = None
        params_to_calibrate = None
        basin_params_to_calibrate = None
        obs_file_path = None
        sim_reach_ID = None
        moo_num_iter = None
        nsga2_n_gen = None
        nsga2_n_obj = None
        filemanager_name = None
        mizu_control_file = None
        optimization_metric = None
        algorithm = None
        optimization_metrics = None
        pop_size = None
        num_iter = None
        calib_period = None
        eval_period = None
        local_bounds_dict = None
        basin_bounds_dict = None
        local_bounds = None
        basin_bounds = None
        all_bounds = None
        all_params = None
        poplsize = None
        swrmsize = None
        ngsize = None
        dds_r = None
        diagnostic_frequency = None
        snow_processed_path = None
        snow_processed_name = None
        snow_station_shapefile_path = None
        snow_station_shapefile_name = None
        MODIS_ndsi_threshold = None
        catchment_shp_name = None
        ostrich_path = None
        ostrich_exe = None
        basin_parameters_file = None
        basin_parameters_file_name = None
        local_parameters_file_name = None
        trial_param_file = None
        settings_summa_attributes = None
        settings_summa_output = None
        settings_summa_filemanager = None
        exe_name_summa = None
        output_path = None
        output_prefix = None
        root_code_path = None
        exe_name_mizuroute = None
        use_mpi = None
        sce_budget = None
        sce_loop_stagnation = None
        sce_pct_change = None
        sce_pop_conv = None
        sce_num_complexes = None
        sce_points_per_complex = None
        sce_points_per_subcomplex = None
        sce_num_evolution_steps = None
        sce_min_complexes = None
        sce_use_initial_point = None
        pso_swarm_size = None
        pso_num_generations = None
        pso_constriction_factor = None
        pso_cognitive_param = None
        pso_social_param = None
        pso_inertia_weight = None
        pso_inertia_reduction_rate = None
        pso_init_population_method = None
        pso_convergence_val = None
        num_ranks = None


    config = Config(
        root_path=comm.bcast(root_path, root=0),
        domain_name=comm.bcast(domain_name, root=0),
        experiment_id=comm.bcast(experiment_id, root=0),
        params_to_calibrate=comm.bcast(params_to_calibrate, root=0),
        basin_params_to_calibrate=comm.bcast(basin_params_to_calibrate, root=0),
        obs_file_path=comm.bcast(obs_file_path, root=0),
        sim_reach_ID=comm.bcast(sim_reach_ID, root=0),
        filemanager_name=comm.bcast(filemanager_name, root=0),
        mizu_control_file=comm.bcast(mizu_control_file, root=0),
        optimization_metric=comm.bcast(optimization_metric, root=0),
        algorithm=comm.bcast(algorithm, root=0),
        num_iter=comm.bcast(num_iter, root=0),
        moo_num_iter=comm.bcast(moo_num_iter, root=0),
        nsga2_n_gen=comm.bcast(nsga2_n_gen, root=0),
        nsga2_n_obj=comm.bcast(nsga2_n_obj, root=0),
        optimization_metrics=comm.bcast(optimization_metrics, root=0),
        pop_size=comm.bcast(pop_size, root=0),
        calib_period=comm.bcast(calib_period, root=0),
        eval_period=comm.bcast(eval_period, root=0),
        local_bounds_dict=comm.bcast(local_bounds_dict, root=0),
        basin_bounds_dict=comm.bcast(basin_bounds_dict, root=0),
        local_bounds=comm.bcast(local_bounds, root=0),
        basin_bounds=comm.bcast(basin_bounds, root=0),
        all_bounds=comm.bcast(all_bounds, root=0),
        all_params=comm.bcast(all_params, root=0),
        poplsize=comm.bcast(poplsize, root=0),
        swrmsize=comm.bcast(swrmsize, root=0),
        ngsize=comm.bcast(ngsize, root=0),
        dds_r=comm.bcast(dds_r, root=0),
        diagnostic_frequency=comm.bcast(diagnostic_frequency,root=0),
        snow_processed_path=comm.bcast(snow_processed_path, root=0),
        snow_processed_name=comm.bcast(snow_processed_name, root=0),
        snow_station_shapefile_path=comm.bcast(snow_station_shapefile_path, root=0),
        snow_station_shapefile_name=comm.bcast(snow_station_shapefile_name, root=0),
        MODIS_ndsi_threshold=comm.bcast(MODIS_ndsi_threshold, root=0),
        catchment_shp_name=comm.bcast(catchment_shp_name, root=0),
        ostrich_path=comm.bcast(ostrich_path, root=0),
        ostrich_exe=comm.bcast(ostrich_exe, root=0),
        ostrich_algorithm=comm.bcast(ostrich_algorithm, root=0),
        ostrich_metric=comm.bcast(ostrich_metric, root=0),
        ostrich_swarm_size=comm.bcast(ostrich_swarm_size, root=0),
        ostrich_output_file=comm.bcast(ostrich_output_file, root=0),
        basin_parameters_file=comm.bcast(basin_parameters_file, root=0),
        basin_parameters_file_name=comm.bcast(basin_parameters_file_name,root=0),
        local_parameters_file_name=comm.bcast(local_parameters_file_name,root=0),
        trial_param_file=comm.bcast(trial_param_file,root=0),
        settings_summa_attributes=comm.bcast(settings_summa_attributes,root=0),
        settings_summa_output=comm.bcast(settings_summa_output,root=0),
        settings_summa_filemanager=comm.bcast(settings_summa_filemanager,root=0),
        exe_name_summa=comm.bcast(exe_name_summa,root=0),
        output_path=comm.bcast(output_path,root=0),
        output_prefix=comm.bcast(output_prefix,root=0),
        root_code_path=comm.bcast(root_code_path,root=0),
        exe_name_mizuroute=comm.bcast(exe_name_mizuroute,root=0),
        use_mpi = comm.bcast(use_mpi,root=0),
        sce_budget = comm.bcast(sce_budget,root=0),
        sce_loop_stagnation = comm.bcast(sce_loop_stagnation,root=0),
        sce_pct_change = comm.bcast(sce_pct_change,root=0),
        sce_pop_conv = comm.bcast(sce_pop_conv,root=0),
        sce_num_complexes = comm.bcast(sce_num_complexes,root=0),
        sce_points_per_complex = comm.bcast(sce_points_per_complex,root=0),
        sce_points_per_subcomplex = comm.bcast(sce_points_per_subcomplex,root=0),
        sce_num_evolution_steps = comm.bcast(sce_num_evolution_steps,root=0),
        sce_min_complexes = comm.bcast(sce_min_complexes,root=0),
        sce_use_initial_point = comm.bcast(sce_use_initial_point,root=0),
        pso_swarm_size = comm.bcast(pso_swarm_size,root=0),
        pso_num_generations = comm.bcast(pso_num_generations,root=0),
        pso_constriction_factor = comm.bcast(pso_constriction_factor,root=0),
        pso_cognitive_param = comm.bcast(pso_cognitive_param,root=0),
        pso_social_param = comm.bcast(pso_social_param,root=0),
        pso_inertia_weight = comm.bcast(pso_inertia_weight,root=0),
        pso_inertia_reduction_rate = comm.bcast(pso_inertia_reduction_rate,root=0),
        pso_init_population_method = comm.bcast(pso_init_population_method,root=0),
        pso_convergence_val = comm.bcast(pso_convergence_val,root=0),
        num_ranks = comm.bcast(num_ranks,root=0)
    )

    return config

@dataclass
class preConfig:
    root_path: Path
    domain_name: str
    catchment_shp_path: str
    river_network_shp_path: str
    river_basin_shp_path: str
    river_basin_shp_name: str
    pour_point_shp_path: str
    source_control_file: str
    pour_point_coords: str
    domain_subsetting:str
    full_domain_name: str
    experiment_id: str
    parameter_dem_tif_name: str
    stream_order_threshold: int
    fullDom_basins_name: str
    fullDom_rivers_name: str
    elevation_band_size: float
    min_hru_size: float
    radiation_class_number: int
    domain_discretisation: str
    pour_point_shp_name: str
    catchment_shp_name: str
    parameter_soil_tif_name: str
    root_code_path: str
    parameter_land_tif_name: str
    datatool_account: str
    forcing_raw_time: str
    num_land_cover: int 
    minimume_land_fraction: float
    num_soil_type: int                
    unify_soil: bool                               
    soil_mLayerDepth: str             
    snow_processed_path: Path
    snow_processed_name: str
    modis_ndsi_threshold: int
    gistool_lib_path: str
    gistool_cache: str
    easymore_cache: str
    datatool_cache: str
    forcing_dataset: str
    gistool_account: str
    datatool_dataset_root: str
    gistool_dataset_root: str
    river_basin_shp_rm_hruid: str
    river_network_shp_name: str
    frac_threshold: float
    write_mizuroute_domain: str
    hru_discr: str
    model: str
 

    @classmethod
    def from_control_file(cls, control_file: Path):
        def read_from_control(file, setting):
            with open(file) as contents:
                for line in contents:
                    if setting in line and not line.startswith('#'):
                        return line.split('|', 1)[1].split('#', 1)[0].strip()
            return None

        root_path = Path(read_from_control(control_file, 'root_path'))
        domain_name = read_from_control(control_file, 'domain_name')
        catchment_shp_path = read_from_control(control_file, 'catchment_shp_path')
        catchment_shp_name = read_from_control(control_file, 'catchment_shp_name')
        river_network_shp_path = read_from_control(control_file, 'river_network_shp_path')
        river_basin_shp_path = read_from_control(control_file, 'river_basin_shp_path')
        pour_point_shp_path = read_from_control(control_file, 'pour_point_shp_path')
        source_control_file = control_file.name
        pour_point_coords = read_from_control(control_file, 'pour_point_coords')
        domain_subsetting = read_from_control(control_file, 'domain_subsetting')
        full_domain_name = read_from_control(control_file, 'full_domain_name')
        experiment_id = read_from_control(control_file, 'experiment_id')
        parameter_dem_tif_name = read_from_control(control_file, 'parameter_dem_tif_name')
        stream_order_threshold = int(read_from_control(control_file, 'stream_order_threshold'))
        fullDom_basins_name = read_from_control(control_file, 'fullDom_basins_name')
        fullDom_rivers_name = read_from_control(control_file, 'fullDom_rivers_name')
        elevation_band_size = int(read_from_control(control_file, 'elevation_band_size'))
        min_hru_size = float(read_from_control(control_file, 'min_hru_size'))
        radiation_class_number = int(read_from_control(control_file, 'radiation_class_number'))
        domain_discretisation = read_from_control(control_file, 'domain_discretisation')
        river_basin_shp_name = read_from_control(control_file, 'river_basin_shp_name')
        pour_point_shp_name = read_from_control(control_file, 'pour_point_shp_name')
        parameter_soil_tif_name = read_from_control(control_file, 'parameter_soil_tif_name')
        root_code_path = read_from_control(control_file, 'root_code_path')
        parameter_land_tif_name = read_from_control(control_file, 'parameter_land_tif_name')
        datatool_account = read_from_control(control_file, 'datatool_account')
        forcing_raw_time = read_from_control(control_file, 'forcing_raw_time')
        num_land_cover = int(read_from_control(control_file, 'num_land_cover'))
        minimume_land_fraction = float(read_from_control(control_file, 'minimume_land_fraction'))
        num_soil_type = int(read_from_control(control_file, 'num_soil_type'))
        unify_soil = bool(read_from_control(control_file, 'unify_soil'))
        soil_mLayerDepth = read_from_control(control_file, 'soil_mLayerDepth')
        snow_processed_path = read_from_control(control_file, 'snow_processed_path')
        snow_processed_name = read_from_control(control_file, 'snow_processed_name')
        modis_ndsi_threshold = int(read_from_control(control_file, 'modis_ndsi_threshold'))
        gistool_lib_path = read_from_control(control_file, 'gistool_lib_path')
        gistool_cache = read_from_control(control_file, 'gistool_cache')
        easymore_cache = read_from_control(control_file, 'easymore_cache')
        datatool_cache = read_from_control(control_file, 'datatool_cache')
        forcing_dataset = read_from_control(control_file, 'forcing_dataset')
        gistool_account = read_from_control(control_file, 'gistool_account')
        datatool_dataset_root = read_from_control(control_file, 'datatool_dataset_root')
        gistool_dataset_root = read_from_control(control_file, 'gistool_dataset_root')
        river_basin_shp_rm_hruid = read_from_control(control_file, 'river_basin_shp_rm_hruid')
        river_network_shp_name = read_from_control(control_file, 'river_network_shp_name')
        frac_threshold = float(read_from_control(control_file, 'frac_threshold'))
        write_mizuroute_domain = read_from_control(control_file, 'write_mizuroute_domain')
        hru_discr = read_from_control(control_file, 'hru_discr')
        model = read_from_control(control_file, 'hru_discr')

        return cls(
            root_path=root_path,
            domain_name=domain_name,
            full_domain_name=full_domain_name,
            experiment_id=experiment_id,
            catchment_shp_path=catchment_shp_path,
            river_network_shp_path=river_network_shp_path,
            river_basin_shp_path=river_basin_shp_path,
            pour_point_shp_path=pour_point_shp_path,
            source_control_file=source_control_file,
            pour_point_coords=pour_point_coords,
            domain_subsetting=domain_subsetting,
            parameter_dem_tif_name=parameter_dem_tif_name,
            stream_order_threshold=stream_order_threshold,
            fullDom_basins_name=fullDom_basins_name,
            fullDom_rivers_name=fullDom_rivers_name,
            elevation_band_size=elevation_band_size,
            min_hru_size=min_hru_size,
            radiation_class_number=radiation_class_number,
            domain_discretisation=domain_discretisation, 
            river_basin_shp_name=river_basin_shp_name,
            pour_point_shp_name=pour_point_shp_name,
            catchment_shp_name=catchment_shp_name,
            parameter_soil_tif_name=parameter_soil_tif_name,
            root_code_path=root_code_path,
            parameter_land_tif_name=parameter_land_tif_name,
            datatool_account=datatool_account,
            forcing_raw_time=forcing_raw_time,
            num_land_cover=num_land_cover,
            minimume_land_fraction=minimume_land_fraction,
            num_soil_type=num_soil_type,
            unify_soil=unify_soil,
            frac_threshold=frac_threshold,
            soil_mLayerDepth=soil_mLayerDepth,
            snow_processed_path=snow_processed_path,
            snow_processed_name=snow_processed_name,
            modis_ndsi_threshold=modis_ndsi_threshold,
            gistool_lib_path=gistool_lib_path,
            gistool_cache=gistool_cache,
            easymore_cache=easymore_cache,
            forcing_dataset=forcing_dataset,
            datatool_cache=datatool_cache,
            gistool_account=gistool_account,
            datatool_dataset_root=datatool_dataset_root,
            gistool_dataset_root=gistool_dataset_root,
            river_basin_shp_rm_hruid=river_basin_shp_rm_hruid,
            river_network_shp_name=river_network_shp_name,
            write_mizuroute_domain=write_mizuroute_domain,
            hru_discr=hru_discr,
            model=model

        )
   