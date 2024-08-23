



def get_features_string(gw_features):
    names = []
    for feature in gw_features:
        if "kriging" in feature:
            names.append(feature.split('kriging_output_')[1].split('_250m')[0])
        else:
            names.append(feature.split('_250m')[0]) 
    
    ## make one string
    names = ' & '.join(names)
    print(names)
    return names


if __name__ == '__main__':
    gw_features_options = ['DEM_250m','kriging_output_SWL_250m',
                           'gSURRGO_swat_250m',  
                           'Aquifer_Characteristics_Of_Glacial_Drift_250m', 'MI_geol_poly_250m', 'landforms_250m_250Dis',
                           'kriging_output_V_COND_1_250m', 'kriging_output_V_COND_2_250m',
                           'kriging_output_AQ_THK_1_250m', 'kriging_output_AQ_THK_2_250m',
                           'kriging_output_H_COND_1_250m', 'kriging_output_H_COND_2_250m',
                           'kriging_output_TRANSMSV_1_250m', 'kriging_output_TRANSMSV_2_250m',
                           "lat_250m", "lon_250m", 
                           "LC22_EVH_220_250m", 
                         'snow_water_equivalent_raster_250m', 
                           "ppt_2018_250m", "ppt_2019_250m", 
                           "ppt_2020_250m", "ppt_2021_250m", "ppt_2022_250m",  
                           "QAMA_MILP_250m",'QBMA_MILP_250m', 'QCMA_MILP_250m', 'QDMA_MILP_250m', "PETMA_MILP_250m",
                           'ArQNavMA_MILP_250m', 'AvgQAdjMA_MILP_250m','COUNTY_250m'
                        ]
    get_features_string(gw_features_options)