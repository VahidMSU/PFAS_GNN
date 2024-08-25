import numpy as np
import torch
from torch_geometric.data import HeteroData
import itertools
import torch



def calculate_distances_gpu(des_coords, src_coords, device, logger, dtype=torch.float):
    # Move tensors to the device and convert to the desired precision
    src_tensor = torch.tensor(src_coords, dtype=dtype, device=device)
    des_tensor = torch.tensor(des_coords, dtype=dtype, device=device)
    
    # Compute distances
    src_des_dists = torch.cdist(src_tensor, des_tensor, p=2)
    
    logger.info(f"stage:calculate_distances_gpu ==##== Shape of src_des_dists: {src_des_dists.shape}")
    logger.info(f"stage:calculate_distances_gpu ==##== Distances shape: {src_des_dists.shape}")
    
    return src_des_dists

def create_edges_and_distances(des_node, src_node, device, threshold, logger):
    assert len(des_node) > 0, logger.error("There are no des_node wells")
    assert len(src_node) > 0, logger.error("There are no src_node")
    assert "geometry" in des_node.columns, logger.error("des_node does not have geometry column")
    assert "geometry" in src_node.columns, logger.error("src_node does not have geometry column")
    assert des_node.crs.to_epsg() != 4326, logger.error("des_node crs is not projected")
    assert src_node.crs.to_epsg() != 4326, logger.error("src_node crs is not projected")

    des_node_coords = np.array([(geom.x, geom.y) for geom in des_node.geometry])
    src_node_coords = np.array([(geom.x, geom.y) for geom in src_node.geometry])

    src_node_des_node_dists = calculate_distances_gpu(des_node_coords, src_node_coords, device, logger)

    edges = []
    distances = []
    dem_differences = []
    swl_differences = []
    num_des_node = len(des_node_coords)
    num_src_node = len(src_node_coords)

    for i, j in itertools.product(range(num_src_node), range(num_des_node)):
        distance = src_node_des_node_dists[i, j].item()
        if distance <= threshold:
            edges.append([i, j])
            distances.append(distance)
            if "DEM_250m" in src_node.columns and "DEM_250m" in des_node.columns:
                dem_difference = src_node['DEM_250m'].iloc[i] - des_node['DEM_250m'].iloc[j]
                dem_differences.append(dem_difference)

            if "kriging_output_SWL_250m" in src_node.columns and "kriging_output_SWL_250m" in des_node.columns:
                fit_to_meter = 0.3048
                site_head = src_node['DEM_250m'].iloc[i] - (src_node['kriging_output_SWL_250m'].iloc[i]*fit_to_meter)
                des_node_head = des_node['DEM_250m'].iloc[j] - (des_node['kriging_output_SWL_250m'].iloc[j]*fit_to_meter)
                swl_difference = site_head - des_node_head
                swl_differences.append(swl_difference)

    logger.info(f'stage:create_edges_and_distances ==##== Number of edges: {len(edges)}')
    logger.info(f'stage:create_edges_and_distances ==##== Number of distances: {len(distances)}')

    ### assert all des_node wells are connected to at least one site
    logger.info(
        f'stage:create_edges_and_distances ==##== Number of des_node wells not connected to at least one src_node {num_des_node - len({edge[1] for edge in edges})}'
    )
    #assert len(set([edge[1] for edge in edges])) == num_des_node, f"{num_des_node - len(set([edge[1] for edge in edges]))} des_node wells are not connected to at least one site"

    return edges, distances, dem_differences, swl_differences

def create_hetdata(pfas_gw, pfas_sw, unsampled_gw, pfas_sites, device, pfas_gw_columns, pfas_sites_columns,pfas_sw_station_columns, gw_features, distance_threshold, logger, gw_gw_distance_threshold):
    
    logger.info(f"stage:create_hetdata ==##== distance_threshold: {distance_threshold}")
    logger.info(f"stage:create_hetdata ==##== Number of pfas_gw: {len(pfas_gw)}")
    logger.info(f"stage:create_hetdata ==##== Number of unsampled gw: {len(unsampled_gw)}")
    logger.info(f"stage:create_hetdata ==##== Number of pfas_sites: {len(pfas_sw)}")


    pfas_gw_columns.append('gw_node_index')
    pfas_sites_columns.append('site_node_index')
    pfas_sw_station_columns.append('sw_node_index')
    data = HeteroData()

    ### normalize gw_features data in pfas_gw, pfas_sw and pfas_sites based on the entire dataset (pfas_gw, pfas_sw and pfas_sites)
    for feature in gw_features:
        all_data_mean = max (np.mean(pfas_gw[feature].values), np.mean(pfas_sw[feature].values), np.mean(pfas_sites[feature].values))
        all_data_std = max (np.std(pfas_gw[feature].values), np.std(pfas_sw[feature].values), np.std(pfas_sites[feature].values))
        pfas_gw[feature] = (pfas_gw[feature] - all_data_mean) / all_data_std
        pfas_sw[feature] = (pfas_sw[feature] - all_data_mean) / all_data_std
        pfas_sites[feature] = (pfas_sites[feature] - all_data_mean) / all_data_std


    data['sw_stations'].x = torch.tensor(pfas_sw[pfas_sw_station_columns + gw_features].values, dtype=torch.float)
    data['gw_wells'].x = torch.tensor(pfas_gw[pfas_gw_columns + gw_features].values, dtype=torch.float)
    data['pfas_sites'].x = torch.tensor(pfas_sites[pfas_sites_columns + gw_features].values, dtype=torch.float)

    

    pfas_gw = pfas_gw.drop_duplicates(subset='WSSN')  ### NOTE: need to be verified later
    #pfas_sw = pfas_sw.drop_duplicates(subset='SiteCode')  ### NOTE: need to be verified later

    assert pfas_gw.crs.to_epsg() == 26990, "pfas_gw crs is not 26990"
    assert pfas_sites.crs.to_epsg() == 26990, "pfas_sites crs is not 26990"
    assert len(pfas_gw['WSSN'].unique()) == len(pfas_gw), f"There are duplicate WSSN values in pfas_gw {len(pfas_gw['WSSN'].unique())} - {len(pfas_gw)}"
    assert len(pfas_sw['SiteCode'].unique()) == len(pfas_sw), f"There are duplicate SiteCode values in pfas_sw {len(pfas_sw['SiteCode'].unique())} - {len(pfas_sw)}"

    
    data = create_node_edge_main(data, pfas_gw, pfas_sw, pfas_sites, device, distance_threshold, logger, gw_gw_distance_threshold)

    data = add_self_loops(data, device)

    assert_HeteroData_creation(data, logger)


    return data



def create_node_edge_main(data, pfas_gw, pfas_sw, pfas_sites, device, distance_threshold, logger, gw_gw_distance_threshold):

    """" Create edges and distances between nodes """

    gw_site_edges, gw_distances, gw_site_dem_differences, gw_site_swl_differences = create_edges_and_distances(pfas_gw, pfas_sites, device, threshold=distance_threshold, logger=logger)
    gw_edge_index = torch.tensor(gw_site_edges, dtype=torch.long).t().contiguous()
    data['pfas_sites', 'dis_edge', 'gw_wells'].edge_index = gw_edge_index
    data['gw_wells', 'dis_edge', 'pfas_sites'].edge_index = gw_edge_index[[1, 0], :]  # Reverse the edges for the reverse direction

    sw_site_edges, sw_distances, sw_site_dem_differences, sw_site_swl_differences = create_edges_and_distances(pfas_sw, pfas_sites, device, threshold=distance_threshold, logger=logger)
    sw_edge_index = torch.tensor(sw_site_edges, dtype=torch.long).t().contiguous()
    data['pfas_sites', 'dis_edge', 'sw_stations'].edge_index = sw_edge_index
    data['sw_stations', 'dis_edge', 'pfas_sites'].edge_index = sw_edge_index[[1, 0], :]  # Reverse the edges for the reverse direction

    gw_sw_edges, gw_sw_distances, gw_sw_dem_differences, gw_sw_swl_differences = create_edges_and_distances(pfas_gw, pfas_sw, device, threshold=distance_threshold, logger=logger)
    gw_sw_edge_index = torch.tensor(gw_sw_edges, dtype=torch.long).t().contiguous()
    data['sw_stations', 'dis_edge', 'gw_wells'].edge_index = gw_sw_edge_index
    data['gw_wells', 'dis_edge', 'sw_stations'].edge_index = gw_sw_edge_index[[1, 0], :]  # Reverse the edges for the reverse direction

    gw_gw_edges, gw_gw_distances, gw_gw_dem_differences, gw_gw_swl_differences = create_edges_and_distances(pfas_gw, pfas_gw, device, threshold=gw_gw_distance_threshold, logger=logger)
    gw_gw_edge_index = torch.tensor(gw_gw_edges, dtype=torch.long).t().contiguous()
    data['gw_wells', 'dis_edge', 'gw_wells'].edge_index = gw_gw_edge_index

    if len(gw_site_dem_differences) > 0 and len(gw_site_swl_differences) == 0:
        logger.info("#### DEM difference is included ##########")
        gw_site_edge_attr = torch.tensor(np.vstack((gw_distances, gw_site_dem_differences)).T, dtype=torch.float)
        sw_site_edge_attr = torch.tensor(np.vstack((sw_distances, sw_site_dem_differences)).T, dtype=torch.float)
        gw_sw_edge_attr = torch.tensor(np.vstack((gw_sw_distances, gw_sw_dem_differences)).T, dtype=torch.float)
        gw_gw_edge_attr = torch.tensor(np.vstack((gw_gw_distances, gw_gw_dem_differences)).T, dtype=torch.float)    
    elif len(gw_site_swl_differences) > 0 and len(gw_site_dem_differences) == 0:
        logger.info("#### SWL difference is included ##########")
        gw_site_edge_attr = torch.tensor(np.vstack((gw_distances, gw_site_swl_differences)).T, dtype=torch.float)
        sw_site_edge_attr = torch.tensor(np.vstack((sw_distances, sw_site_swl_differences)).T, dtype=torch.float)
        gw_sw_edge_attr = torch.tensor(np.vstack((gw_sw_distances, gw_sw_swl_differences)).T, dtype=torch.float)
        gw_gw_edge_attr = torch.tensor(np.vstack((gw_gw_distances, gw_gw_swl_differences)).T, dtype=torch.float)

    elif len(gw_site_dem_differences) > 0 and len(gw_site_swl_differences) > 0:
        logger.info("#### DEM difference and SWL difference is included ##########")
        gw_site_edge_attr = torch.tensor(np.vstack((gw_distances, gw_site_dem_differences, gw_site_swl_differences)).T, dtype=torch.float)
        sw_site_edge_attr = torch.tensor(np.vstack((sw_distances, sw_site_dem_differences, sw_site_swl_differences)).T, dtype=torch.float)
        gw_sw_edge_attr = torch.tensor(np.vstack((gw_sw_distances, gw_sw_dem_differences, gw_sw_swl_differences)).T, dtype=torch.float)
        gw_gw_edge_attr = torch.tensor(np.vstack((gw_gw_distances, gw_gw_dem_differences, gw_gw_swl_differences)).T, dtype=torch.float)
    else:
        gw_site_edge_attr = torch.tensor(np.vstack((gw_distances)).T, dtype=torch.float)
        sw_site_edge_attr = torch.tensor(np.vstack((sw_distances)).T, dtype=torch.float)
        gw_sw_edge_attr = torch.tensor(np.vstack((gw_sw_distances)).T, dtype=torch.float)
        gw_gw_edge_attr = torch.tensor(np.vstack((gw_gw_distances)).T, dtype=torch.float)

    data['pfas_sites', 'dis_edge', 'gw_wells'].edge_attr = gw_site_edge_attr
    data['gw_wells', 'dis_edge', 'pfas_sites'].edge_attr = gw_site_edge_attr

    data['pfas_sites', 'dis_edge', 'sw_stations'].edge_attr = sw_site_edge_attr
    data['sw_stations', 'dis_edge', 'pfas_sites'].edge_attr = sw_site_edge_attr

    data['sw_stations', 'dis_edge', 'gw_wells'].edge_attr = gw_sw_edge_attr
    data['gw_wells', 'dis_edge', 'sw_stations'].edge_attr = gw_sw_edge_attr

    data['gw_wells', 'dis_edge', 'gw_wells'].edge_attr = gw_gw_edge_attr
    data['gw_wells', 'dis_edge', 'gw_wells'].edge_attr = gw_gw_edge_attr

    return data

def assert_HeteroData_creation(data, logger):
    # Ensure 'pfas_sites' is a destination node type
    assert ('pfas_sites', 'dis_edge', 'gw_wells') in data.edge_index_dict.keys(), "There is no edge from 'pfas_sites' to 'gw_wells'"
    assert ('gw_wells', 'dis_edge', 'pfas_sites') in data.edge_index_dict.keys(), "There is no edge from 'gw_wells' to 'pfas_sites'"

    assert ('pfas_sites', 'self_loop', 'pfas_sites') in data.edge_index_dict.keys(), "There is no self-loop edge for 'pfas_sites'"
    assert ('gw_wells', 'self_loop', 'gw_wells') in data.edge_index_dict.keys(), "There is no self-loop edge for 'gw_wells'"

    assert ('pfas_sites', 'dis_edge', 'sw_stations') in data.edge_index_dict.keys(), "There is no edge from 'pfas_sites' to 'sw_stations'"
    assert ('sw_stations', 'dis_edge', 'pfas_sites') in data.edge_index_dict.keys(), "There is no edge from 'sw_stations' to 'pfas_sites'"

    assert ('pfas_sites', 'self_loop', 'pfas_sites') in data.edge_index_dict.keys(), "There is no self-loop edge for 'pfas_sites'"
    assert ('sw_stations', 'self_loop', 'sw_stations') in data.edge_index_dict.keys(), "There is no self-loop edge for 'sw_stations'"
    
    assert ('sw_stations', 'dis_edge', 'gw_wells') in data.edge_index_dict.keys(), "There is no edge from 'sw_stations' to 'gw_wells'"
    assert ('gw_wells', 'dis_edge', 'sw_stations') in data.edge_index_dict.keys(), "There is no edge from 'gw_wells' to 'sw_stations'"
    assert ('sw_stations', 'self_loop', 'sw_stations') in data.edge_index_dict.keys(), "There is no self-loop edge for 'sw_stations'"

    logger.info(f"Edges from 'pfas_sites' to 'sw_stations': {data['pfas_sites', 'dis_edge', 'sw_stations'].edge_index.shape[1]}")
    logger.info(f"Edges from 'sw_stations' to 'pfas_sites': {data['sw_stations', 'dis_edge', 'pfas_sites'].edge_index.shape[1]}")

    logger.info(f"Edges from 'pfas_sites' to 'gw_wells': {data['pfas_sites', 'dis_edge', 'gw_wells'].edge_index.shape[1]}")
    logger.info(f"Edges from 'gw_wells' to 'pfas_sites': {data['gw_wells', 'dis_edge', 'pfas_sites'].edge_index.shape[1]}")

    logger.info(f"Edges from 'sw_stations' to 'gw_wells': {data['sw_stations', 'dis_edge', 'gw_wells'].edge_index.shape[1]}")
    logger.info(f"Edges from 'gw_wells' to 'sw_stations': {data['gw_wells', 'dis_edge', 'sw_stations'].edge_index.shape[1]}")

    logger.info(f"Self-loop edges for 'pfas_sites': {data['pfas_sites', 'self_loop', 'pfas_sites'].edge_index.shape[1]}")
    logger.info(f"Self-loop edges for 'gw_wells': {data['gw_wells', 'self_loop', 'gw_wells'].edge_index.shape[1]}")
    logger.info(f"Self-loop edges for 'sw_stations': {data['sw_stations', 'self_loop', 'sw_stations'].edge_index.shape[1]}")


def add_self_loops(data, device):
    """Add self-loops to the HeteroData."""
    num_pfas_sites = data['pfas_sites'].x.size(0)
    num_gw_wells = data['gw_wells'].x.size(0)
    num_sw_stations = data['sw_stations'].x.size(0)

    pfas_sites_self_loops = torch.arange(num_pfas_sites, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
    gw_wells_self_loops = torch.arange(num_gw_wells, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
    sw_stations_self_loops = torch.arange(num_sw_stations, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)

    data['pfas_sites', 'self_loop', 'pfas_sites'].edge_index = pfas_sites_self_loops
    data['gw_wells', 'self_loop', 'gw_wells'].edge_index = gw_wells_self_loops
    data['sw_stations', 'self_loop', 'sw_stations'].edge_index = sw_stations_self_loops

    return data


def create_rivers_nodes_edges():
    import geopandas as gpd 
    import pandas as pd
    path = "/data/MyDataBase/CIWRE-BAE/SWAT_input/huc8/4100013/SWAT_MODEL/Watershed/Shapes/rivs1.shp"
    sw = gpd.read_file(path)    ##LINKNO, DSLINKNO
    sw['riv_node_index'] = pd.factorize(sw['LINKNO'])[0]
    ### define edges by LINKNO and DSLINKNO relationship 
    

    sw_edges = sw[['LINKNO', 'DSLINKNO']].values
    sw_edges = np.unique(sw_edges, axis=0)
    sw_edge_index = torch.tensor(sw_edges.T, dtype=torch.long)
    ### define edge_attr using relevant columns
    sw_edge_attr = torch.tensor(sw[['LENGTHKM', 'SLOPE', 'MANNING']].values, dtype=torch.float)
    
    
    print(sw.columns)  

if __name__ == "__main__":
    create_rivers_nodes_edges()
    
    print("End of the script")