
# config.py

from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime
from pathlib import Path
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

def parse_bool(value):
    return value.lower() in ('true', 'yes', '1', 'on')

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


control_folder = Path('/Users/darrieythorsson/compHydro/code/CWARHM/0_control_files/')
control_file = 'control_active.txt'



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
    use_parallel_summa: str
    summa_cpus_per_task: str
    summa_time_limit: str 
    summa_mem: str
    summa_gru_count: int   
    summa_gru_per_job: int   

    
    @classmethod
    def from_control_file(cls, control_file: Path):
        def read_from_control(file, setting):
            with open(file) as contents:
                for line in contents:
                    if setting in line and not line.startswith('#'):
                        return line.split('|', 1)[1].split('#', 1)[0].strip()
            return None
        
        
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
        use_parallel_summa = parse_bool(read_from_control(control_folder/control_file, 'use_parallel_summa'))
        summa_cpus_per_task = read_from_control(control_folder/control_file, 'summa_cpus_per_task')
        summa_time_limit = read_from_control(control_folder/control_file, 'summa_time_limit')
        summa_mem = read_from_control(control_folder/control_file, 'summa_mem')
        summa_gru_count = int(read_from_control(control_folder/control_file, 'summa_gru_count'))
        summa_gru_per_job = int(read_from_control(control_folder/control_file, 'summa_gru_per_job'))


        return cls(
            root_path=root_path,
            domain_name=domain_name,
            experiment_id=experiment_id,
            params_to_calibrate=params_to_calibrate,
            basin_params_to_calibrate=basin_params_to_calibrate,
            obs_file_path=obs_file_path,
            sim_reach_ID=sim_reach_ID,
            filemanager_name=filemanager_name,
            mizu_control_file=mizu_control_file,
            optimization_metric=optimization_metric,
            algorithm=algorithm,
            num_iter=num_iter,
            moo_num_iter=moo_num_iter,
            nsga2_n_gen=nsga2_n_gen,
            nsga2_n_obj=nsga2_n_obj,
            optimization_metrics=optimization_metrics,
            pop_size=pop_size,
            calib_period=calib_period,
            eval_period=eval_period,
            local_bounds_dict=local_bounds_dict,
            basin_bounds_dict=basin_bounds_dict,
            local_bounds=local_bounds,
            basin_bounds=basin_bounds,
            all_bounds=all_bounds,
            all_params=all_params,
            poplsize=poplsize,
            swrmsize=swrmsize,
            ngsize=ngsize,
            dds_r=dds_r,
            diagnostic_frequency=diagnostic_frequency,
            snow_processed_path=snow_processed_path,
            snow_processed_name=snow_processed_name,
            snow_station_shapefile_path=snow_station_shapefile_path,
            snow_station_shapefile_name=snow_station_shapefile_name,
            MODIS_ndsi_threshold=MODIS_ndsi_threshold,
            catchment_shp_name=catchment_shp_name,
            ostrich_path=ostrich_path,
            ostrich_exe=ostrich_exe,
            ostrich_algorithm=ostrich_algorithm,
            ostrich_metric=ostrich_metric,
            ostrich_swarm_size=ostrich_swarm_size,
            ostrich_output_file=ostrich_output_file,
            basin_parameters_file=basin_parameters_file,
            basin_parameters_file_name=basin_parameters_file_name,
            local_parameters_file_name=local_parameters_file_name,
            trial_param_file=trial_param_file,
            settings_summa_attributes=settings_summa_attributes,
            settings_summa_output=settings_summa_output,
            settings_summa_filemanager=settings_summa_filemanager,
            exe_name_summa=exe_name_summa,
            output_path=output_path,
            output_prefix=output_prefix,
            root_code_path=root_code_path,
            exe_name_mizuroute=exe_name_mizuroute,
            use_mpi = use_mpi,
            sce_budget = sce_budget,
            sce_loop_stagnation = sce_loop_stagnation,
            sce_pct_change = sce_pct_change,
            sce_pop_conv = sce_pop_conv,
            sce_num_complexes = sce_num_complexes,
            sce_points_per_complex = sce_points_per_complex,
            sce_points_per_subcomplex = sce_points_per_subcomplex,
            sce_num_evolution_steps = sce_num_evolution_steps,
            sce_min_complexes = sce_min_complexes,
            sce_use_initial_point = sce_use_initial_point,
            pso_swarm_size = pso_swarm_size,
            pso_num_generations = pso_num_generations,
            pso_constriction_factor = pso_constriction_factor,
            pso_cognitive_param = pso_cognitive_param,
            pso_social_param = pso_social_param,
            pso_inertia_weight = pso_inertia_weight,
            pso_inertia_reduction_rate = pso_inertia_reduction_rate,
            pso_init_population_method = pso_init_population_method,
            pso_convergence_val = pso_convergence_val,
            num_ranks = num_ranks,
            use_parallel_summa = use_parallel_summa,
            summa_cpus_per_task = summa_cpus_per_task,
            summa_time_limit = summa_time_limit,
            summa_mem = summa_mem,
            summa_gru_count = summa_gru_count,
            summa_gru_per_job = summa_gru_per_job




        )

  
