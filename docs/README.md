## Model and Calibration Configurations

### Data Configurations

- **Meteorology Data:** [RDRS](https://github.com/kasra-keshavarz/datatool/tree/main/scripts/eccc-rdrs) 
  - SUMMA: No temperature lapsing. This differs from the CWARHM configuration.
  
- **Land Cover:** [Landsat NALCMS 2020 Version](https://github.com/kasra-keshavarz/gistool/tree/main/landsat)
  - SUMMA: Convert to the USGS land cover table using Mohamad’s correspondence approach ([here](https://github.com/MIsmlAhmed/MAF/blob/main/03_model_specific_component/03_summa/write_summa_files.ipynb)).

- **Soil Data:** [USDA Soil Classification](https://github.com/kasra-keshavarz/gistool/tree/main/soil_class)  
  - SUMMA: Convert to the ROSETTA soil table.

- **Digital Elevation Model (DEM):** [MERIT-Hydro](https://doi.org/10.1029/2019WR024873)

### Other Configurations

- **Modeling Period:** October 1, 2008 – September 30, 2018  
  - 1-year warm-up: October 1, 2008 – September 30, 2009
  - 5-year calibration: October 1, 2009 – September 30, 2014
  - 4-year validation: October 1, 2014 – September 30, 2018

- **Calibration Tool:** Ostrich

- **Objective Function:** Modified Kling-Gupta Efficiency (KGE)  
  - *Reference:* Kling, Harald, Martin Fuchs, and Maria Paulin. “Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios.” *Journal of Hydrology* 424 (2012): 264-277.

- **Observation:** Daily streamflow downloaded from [HYDAT Database](https://wateroffice.ec.gc.ca/mainmenu/historical_data_index_e.html)
  - Rodolfo has downloaded and shared these data, which are located in `examples/obs` of this repository.
  
