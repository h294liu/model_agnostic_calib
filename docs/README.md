## Model and Calibration Configurations

### Data Configurations

- Meteorology Data: [Regional Deterministic Prediction System (RDPS)](https://github.com/kasra-keshavarz/datatool/tree/main/scripts/eccc-rdrs) 
  
- Land Cover: [Landsat Geospatial Dataset 2020 Version](https://github.com/kasra-keshavarz/gistool/tree/main/landsat)
  
- Soil Data: [USDA Soil Classification](https://hydroshare.org/resource/1361509511e44adfba814f6950c6e742/)  

- Digital Elevation Model (DEM): [MERIT-Hydro](https://doi.org/10.1029/2019WR024873)

### Calibration Configurations

- **Modeling Period:** Oct 1, 2007 – Sept 30, 2018  
  - 1-year warm-up: Oct 1, 2007 – Sept 30, 2008
  - 5-year calibration: Oct 1, 2008 – Sept 30, 2013
  - 5-year validation: Oct 1, 2013 – Sept 30, 2018

- **Calibration Tool:** Ostrich

- **Objective Function:** Modified Kling-Gupta Efficiency (KGE)
  - Modified KGE is expressed as:

    $$
    KGE' = 1 - \sqrt{(r - 1)^2 + (\beta - 1)^2 + (\gamma - 1)^2}
    $$

    $$
    \beta = \frac{\mu_{\text{sim}}}{\mu_{\text{obs}}}
    $$

    $$
    \gamma = \frac{CV_{\text{sim}}}{CV_{\text{obs}}} = \frac{{\sigma_{\text{sim}}}/{\mu_{\text{sim}}}}{{\sigma_{\text{obs}}}/{\mu_{\text{obs}}}}
    $$

    - r is the Pearson correlation coefficient.
    - β (beta) is the bias ratio.
    - γ (gamma) is the variability ratio.

  - The range of KGE' is \(-∞, 1]\. A KGE' value of 1 indicates perfect agreement between the simulated and observed data.

  - Reference: Kling, Harald, Martin Fuchs, and Maria Paulin. “Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios.” *Journal of Hydrology* 424 (2012): 264-277.

- **Observation:** Daily streamflow downloaded from [HYDAT Database](https://wateroffice.ec.gc.ca/mainmenu/historical_data_index_e.html)
  - Example streafmlow data are located in `examples/obs` of this repository.
  
### Notes
 
- **Data Processing Tools:** All meteorological and attribute data have been prepared using the [Modeling Community Workflows](https://github.com/kasra-keshavarz/community-modelling-workflow-training.git). 
   
- For **SUMMA** modelers: 
  - Meteorology data: No temperature lapsing. This differs from the CWARHM configuration.
  - Land Cover: Convert to the USGS land cover table using Dr. Mohamad Ahmed’s correspondence approach ([here](https://github.com/MIsmlAhmed/MAF/blob/main/03_model_specific_component/03_summa/write_summa_files.ipynb)).
  - Soil: Convert to the ROSETTA soil table.
