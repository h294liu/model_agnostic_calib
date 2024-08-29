# load the needed libraries
import numpy  as np
import pandas as pd
import os
import warnings
import shutil


# create the ostin setup (folder structure)
def initialize_ostin_setup(hype_inputs_path, output_path):
    # create the common_inputs directory, which contains all common inputs such as input files, validation files, etc.)
    if not os.path.isdir(os.path.join(output_path,'common_inputs')):
        os.makedirs(os.path.join(output_path,'common_inputs'))
    # create the ost directory    
    if not os.path.isdir(os.path.join(output_path,'ost')):
        os.makedirs(os.path.join(output_path,'ost'))
    # copy input files to common_inpts and ignore the par file, log files, and results directory
    shutil.copytree(hype_inputs_path, os.path.join(output_path,'common_inputs'), dirs_exist_ok=True, ignore= shutil.ignore_patterns('par.txt', 'results*', '*.log'))

    # write the bad_simass.txt file
    bad_simass_file = """Simulation assessment all variables/criterions used
Simulation number:            1
Total criteria value:   -9999.00000    Conditional criteria value:    0.00000000     , threshold:    0.00000000    

Individual criterions
crit  1
-9999.00000    

Variables: rout, cout
Period: DD
Regional NSE:  -9999.00000      
Regional  RA:  -9999.00000    
Regional  RE:  -9999.00000       
Regional MAE:  -9999.00000        
Average  NSE:  -9999.00000      
Average   RA:  -9999.00000    
Average   RE:  -9999.00000    
Average RSDE:  -9999.00000    
Average   CC:  -9999.00000    
Average  ARE:  -9999.00000    
Average  KGE: -9999.00000    
Aver scalKGE: -9999.00000        
Spatial  NSE:  -9999.00000    
Spatial   RA:  -9999.00000    
Spatial   RE:  -9999.00000    
Spatial Bias:  -9999.00000    
Spatial RMSE:  -9999.00000    
Kendalls Tau:  -9999.00000    
Median   NSE:  -9999.00000       
Median    RA:  -9999.00000    
Median   KGE: -9999.00000      
Median NRMSE:  -9999.00000    
Mean    NSEW:  -9999.00000    
Number of data for regional criterion:   -9999.00000    
Number of areas in mean/median criterion:   -9999.00000    

"""
    
    with open(os.path.join(output_path,'common_inputs','bad_simass_file.txt'), 'w') as file:
        file.writelines(bad_simass_file)

########################################################
########################################################

# get the number of instances for each parameter (based on number of landcover types, soil types, ilregion, etc.)
def get_param_instances(hype_inputs_path):
    # initialize dictionary
    param_instances = {}
    # read the GeoClass file
    GeoClass = pd.read_csv(os.path.join(hype_inputs_path,'GeoClass.txt'), sep='\t', comment='!', header=None)
    GeoClass.columns=['SLC',	'LULC',	'SOIL TYPE',	'Main crop cropid',	'Second crop cropid',	'Crop rotation group',	'Vegetation type',	'Special class code',	'Tile depth',	'Stream depth',	'Number of soil layers',	'Soil layer depth 1',	'Soil layer depth 2',	'Soil layer depth 3']
    # get the number of landcover types and soil types in the GeoClass file
    param_instances['landcover'] = len(np.unique(GeoClass['LULC']))
    param_instances['soil'] = len(np.unique(GeoClass['SOIL TYPE']))
    param_instances['special_classes'] = np.count_nonzero(np.unique(GeoClass['Special class code']))
    if(param_instances['special_classes']>0):
        warnings.warn("Warning...........Message")
        print('This HYPE setup contains special classes. \n',
            'These classes require special treatment in the calibration. \n',
            'The current code does not account for that. \n'
            'Make sure to check and fix the optimization setup')

    #########
    # read the GeoData file
    GeoData = pd.read_csv(os.path.join(hype_inputs_path,'GeoData.txt'), sep='\t')
    # initialize ilregion/olregion counter
    param_instances['ilregion'] = param_instances['olregion'] = 0
    # check if ilregion and olregion columns exist in the GeoData file
    if 'ilregion' in GeoData:
        param_instances['ilregion'] = np.count_nonzero(np.unique(GeoData['ilregion']))

    if 'olregion' in GeoData:
        param_instances['olregion'] = np.count_nonzero(np.unique(GeoData['olregion']))

    # general parameters (number of instances = 1)
    param_instances['general'] = 1

    return param_instances

########################################################
########################################################

# read the parameters range file
def read_HYPE_MinMaxParaRange(minmaxParamFile, hype_inputs_path):
    
    # initialize hype parameter range dataframe
    hype_param_range = pd.DataFrame([])

    # get the number of instances for each parameter to link to the dataframe
    nparam_instances = get_param_instances(hype_inputs_path)

    # Open the file and read its contents
    with open(minmaxParamFile, 'r') as file:
        # Read the file line by line
        for line in file:
            # Skip lines that are comments or empty
            if line.startswith('#') or not line.strip():
                continue
            # Split the line into components based on the delimiter '|'
            parts = line.split('|')
            # Extract the parameter name, min, max, and dependency, and strip any extra spaces (ignore description)
            parameter_name = parts[0].strip()
            min = parts[1].strip()
            max = parts[2].strip()
            dependency = parts[3].strip()
            ninstances = nparam_instances[dependency]
            # Store the values in a dataframe
            # Append the row with the singular values
            hype_param_range = pd.concat([hype_param_range, pd.DataFrame({
                'name': [parameter_name],
                'min': [min],
                'max': [max],
                'dependency': [dependency],
                'ninstances': [ninstances]})], ignore_index=True)
    
    # set the name column as the index
    hype_param_range.set_index('name', inplace=True)        
    # convert it to a dictionary for easier mapping and slicing
    hype_param_range = hype_param_range.to_dict(orient='index')

    return(hype_param_range)

########################################################
########################################################

def write_ostin_tpl_files(hype_inputs_path, output_path, hype_exe_path, param_range, optimization_param):
    # write the ostin.txt file 
    # create the parameters name to be fed to the ostin
    ostin_param = pd.DataFrame(columns=['name', 'initVal',	'min', 'max', 'transformations1', 'transformations2', 'transformations3', 'format'])
    # create the parameter section of the ostin file
    # loop through parameters in the param_range dict
    for par_name,par_meta_data in param_range.items():
        if(par_meta_data['ninstances']!=0):
            npar = par_meta_data['ninstances']
            data_to_append = {
                'name'            : [f'__{par_name}__{i:02}__' for i in np.arange(npar)+1],
                'initVal'         :  np.repeat('random',npar),
                'min'             :  np.repeat(par_meta_data['min'],npar),
                'max'             :  np.repeat(par_meta_data['max'],npar),
                'transformations1':  np.repeat('none',npar),
                'transformations2':  np.repeat('none',npar),
                'transformations3':  np.repeat('none',npar),
                'format'          :  np.repeat('F15.8',npar)
                }
            # append the data
            ostin_param = pd.concat([ostin_param, pd.DataFrame(data_to_append)], ignore_index=True)

# write to the ostin file
    ostin_file = """ProgramType  placeholder_algorithm
ObjectiveFunction  GCOP
RandomSeed 123456
ModelExecutable  ./ost_hype.sh
ModelSubdir Processor_

BeginFilePairs
par.tpl ; par.txt
EndFilePairs

BeginParams
#name		        init.	lower	    upper	    transformations         format
placeholder_ostin_params
EndParams

BeginResponseVars
#name	filename		keyword		line	col	token	augment?
placeholder_objFun_name_loc
EndResponseVars

placeholder_tiedVars_GCOP

BeginParallelDDSAlg
PerturbationValue 0.2
MaxIterations placeholder_DDS_niter
EndParallelDDSAlg
"""

# Define a dictionary of placeholders and their replacements
    replacements = {
        'placeholder_algorithm': optimization_param['opt_algorithm'],
        'placeholder_ostin_params': ostin_param.to_csv(sep='\t', index=False, header=None),
        'placeholder_DDS_niter': str(optimization_param['ninter']),
        'placeholder_objFun_name_loc': optimization_param['objFun_name']+'\t'+
                                       optimization_param['objFun_file']+';\t OST_NULL'+'\t'+
                                       str(optimization_param['objFun_row']-1) + '\t' + 
                                       str(optimization_param['objFun_col']) + "\t' ' yes"
    }

    # Replace placeholders in the text content
    for placeholder, replacement in replacements.items():
        ostin_file = ostin_file.replace(placeholder, replacement)


    if optimization_param['objFun_type'] == 'minimize':
        new_content = (
            """
BeginGCOP
CostFunction {objFun_name}
PenaltyFunction APM
EndGCOP"""
        ).format(objFun_name=optimization_param['objFun_name'])


    if optimization_param['objFun_type'] == 'maximize':
        new_content = (
            """BeginTiedRespVars
NEG_{objFun_name} 1 {objFun_name} wsum -1.00
EndTiedRespVars

BeginGCOP
CostFunction NEG_{objFun_name}
PenaltyFunction APM
EndGCOP"""
        ).format(objFun_name=optimization_param['objFun_name'])

    ostin_file = ostin_file.replace('placeholder_tiedVars_GCOP', new_content)
    
    with open(os.path.join(output_path,'ost','ostIn.txt'), 'w') as file:
        file.write(ostin_file)
    
    ###################
    # write the par.tpl file
    # edit the par file
    with open(os.path.join(hype_inputs_path, 'par.txt'), 'r') as file:
        lines = file.readlines()



    # loop over parameters and replace with values
    for par_name,par_meta_data in param_range.items():
            if(par_meta_data['ninstances']!=0):
                npar = par_meta_data['ninstances']
                data_to_append = {
                    'name'            : [par_name]+[f'__{par_name}__{i:02}__' for i in np.arange(npar)+1]
                    }
                
                # Manually create a single tab-delimited string
                data_str = '\t'.join(pd.DataFrame(data_to_append).values.flatten()) + '\n' 
                
                # Replace the line starting with the par_name with the DataFrame content
                lines = [data_str if line.startswith(f"{par_name}\t") else line for line in lines]

    with open(os.path.join(output_path,'ost','par.tpl'), 'w') as file:
        file.writelines(lines)

######################
    # write the ost_hype.sh file
    ost_hype_file = """#!/bin/bash

echo inside ost_hype Script

# specify the common inputs data directory
dir_inputs=../../common_inputs
# specify the HYPE model directory
HYPE_exe=placeholder_hype_path

rm -r *.log
rm -r results/
mkdir results/

pwd
cp -n $HYPE_exe ./hype

ln -s $dir_inputs/*.* ./

# RUN hype
./hype

echo hype finished

# calculate metrics

if [ -f ./results/simass.txt ]; then
	echo 'Good run'
else
  cp bad_simass_file.txt results/simass.txt
  echo 'Bad run'
fi

echo ost_hype Script Finished.

"""
    ost_hype_file = ost_hype_file.replace('placeholder_hype_path', hype_exe_path)
    
    with open(os.path.join(output_path,'ost','ost_hype.sh'), 'w') as file:
        file.writelines(ost_hype_file)

    # Make the shell script executable
    os.chmod(os.path.join(output_path,'ost','ost_hype.sh'), 0o755)
