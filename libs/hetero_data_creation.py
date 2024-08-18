import numpy as np
import torch
from torch_geometric.data import HeteroData
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear, JumpingKnowledge
from torch_geometric.nn import global_mean_pool, global_max_pool




def calculate_distances_gpu(gw_coords, sites_coords, device, logger):
    sites_tensor = torch.tensor(sites_coords, dtype=torch.float, device=device)
    gw_tensor = torch.tensor(gw_coords, dtype=torch.float, device=device)
    sites_gw_dists = torch.cdist(sites_tensor, gw_tensor, p=2)
    logger.info(f"stage:calculate_distances_gpu ==##== Shape of sites_gw_dists: {sites_gw_dists.shape}")
    logger.info(f"stage:calculate_distances_gpu ==##== Distances shape: {sites_gw_dists.shape}")
    return sites_gw_dists

def create_edges_and_distances(gw, sites, device, threshold, logger):
    assert len(gw) > 0, logger.error("There are no gw wells")
    assert len(sites) > 0, logger.error("There are no sites")
    assert gw.crs.to_epsg() != 4326, logger.error("gw crs is not projected")
    assert sites.crs.to_epsg() != 4326, logger.error("sites crs is not projected")
    assert len(gw['WSSN'].unique()) == len(gw), logger.error("There are duplicate WSSN values in gw")

    gw_coords = np.array([(geom.x, geom.y) for geom in gw.geometry])
    sites_coords = np.array([(geom.x, geom.y) for geom in sites.geometry])

    sites_gw_dists = calculate_distances_gpu(gw_coords, sites_coords, device, logger)

    edges = []
    distances = []
    dem_differences = []
    swl_differences = []
    num_gw = len(gw_coords)
    num_sites = len(sites_coords)

    for i, j in itertools.product(range(num_sites), range(num_gw)):
        distance = sites_gw_dists[i, j].item()
        if distance <= threshold:
            edges.append([i, j])
            distances.append(distance)
            if "DEM_250m" in sites.columns and "DEM_250m" in gw.columns:
                dem_difference = sites['DEM_250m'].iloc[i] - gw['DEM_250m'].iloc[j]
                dem_differences.append(dem_difference)

            if "kriging_output_SWL_250m" in sites.columns and "kriging_output_SWL_250m" in gw.columns:
                fit_to_meter = 0.3048
                site_head = sites['DEM_250m'].iloc[i] - (sites['kriging_output_SWL_250m'].iloc[i]*fit_to_meter)
                gw_head = gw['DEM_250m'].iloc[j] - (gw['kriging_output_SWL_250m'].iloc[j]*fit_to_meter)
                swl_difference = site_head - gw_head
                swl_differences.append(swl_difference)

    logger.info(f'stage:create_edges_and_distances ==##== Number of edges: {len(edges)}')
    logger.info(f'stage:create_edges_and_distances ==##== Number of distances: {len(distances)}')

    ### assert all gw wells are connected to at least one site
    logger.info(
        f'stage:create_edges_and_distances ==##== Number of gw wells not connected to at least one sites {num_gw - len({edge[1] for edge in edges})}'
    )
    #assert len(set([edge[1] for edge in edges])) == num_gw, f"{num_gw - len(set([edge[1] for edge in edges]))} gw wells are not connected to at least one site"

    return edges, distances, dem_differences, swl_differences

def create_hetdata(pfas_gw, unsampled_gw, pfas_sites, device, pfas_gw_columns, pfas_sites_columns, gw_features, distance_threshold, logger):
    logger.info(f"stage:create_hetdata ==##== distance_threshold: {distance_threshold}")
    logger.info(f"stage:create_hetdata ==##== Number of pfas_gw: {len(pfas_gw)}")
    logger.info(f"stage:create_hetdata ==##== Number of unsampled gw: {len(unsampled_gw)}")

    pfas_gw_columns.append('gw_node_index')
    pfas_sites_columns.append('site_node_index')
    data = HeteroData()

    data['gw_wells'].x = torch.tensor(pfas_gw[pfas_gw_columns + gw_features].values, dtype=torch.float)
    for col in pfas_sites_columns:
        if pfas_sites[col].dtype == 'O':
            raise ValueError(f"Column {col} is of type object")
    data['pfas_sites'].x = torch.tensor(pfas_sites[pfas_sites_columns + gw_features].values, dtype=torch.float)

    pfas_gw = pfas_gw.drop_duplicates(subset='WSSN')

    assert pfas_gw.crs.to_epsg() == 26990, "pfas_gw crs is not 26990"
    assert pfas_sites.crs.to_epsg() == 26990, "pfas_sites crs is not 26990"
    assert len(pfas_gw['WSSN'].unique()) == len(pfas_gw), f"There are duplicate WSSN values in pfas_gw {len(pfas_gw['WSSN'].unique())} - {len(pfas_gw)}"

    edges, distances, dem_differences, swl_differences = create_edges_and_distances(pfas_gw, pfas_sites, device, threshold=distance_threshold, logger=logger)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data['pfas_sites', 'distance', 'gw_wells'].edge_index = edge_index
    data['gw_wells', 'distance', 'pfas_sites'].edge_index = edge_index[[1, 0], :]  # Reverse the edges for the reverse direction

    if len(dem_differences) > 0 and len(swl_differences) == 0:
        logger.info("#### DEM difference is included ##########")
        edge_attr = torch.tensor(np.vstack((distances, dem_differences)).T, dtype=torch.float)
        data['pfas_sites', 'distance', 'gw_wells'].edge_attr = edge_attr
        data['gw_wells', 'distance', 'pfas_sites'].edge_attr = edge_attr
    elif len(swl_differences) > 0 and len(dem_differences) == 0:
        logger.info("#### SWL difference is included ##########")
        edge_attr = torch.tensor(np.vstack((distances, swl_differences)).T, dtype=torch.float)
        data['pfas_sites', 'distance', 'gw_wells'].edge_attr = edge_attr
        data['gw_wells', 'distance', 'pfas_sites'].edge_attr = edge_attr
    elif len(dem_differences) > 0 and len(swl_differences) > 0:
        logger.info("#### DEM difference and SWL difference is included ##########")
        edge_attr = torch.tensor(np.vstack((distances, dem_differences, swl_differences)).T, dtype=torch.float)
        data['pfas_sites', 'distance', 'gw_wells'].edge_attr = edge_attr
        data['gw_wells', 'distance', 'pfas_sites'].edge_attr = edge_attr
    else:
        edge_attr = torch.tensor(np.vstack((distances)).T, dtype=torch.float)
        data['pfas_sites', 'distance', 'gw_wells'].edge_attr = edge_attr
        data['gw_wells', 'distance', 'pfas_sites'].edge_attr = edge_attr

    # Add self-loops
    num_pfas_sites = data['pfas_sites'].x.size(0)
    num_gw_wells = data['gw_wells'].x.size(0)

    pfas_sites_self_loops = torch.arange(num_pfas_sites, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
    gw_wells_self_loops = torch.arange(num_gw_wells, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)

    data['pfas_sites', 'self_loop', 'pfas_sites'].edge_index = pfas_sites_self_loops
    data['gw_wells', 'self_loop', 'gw_wells'].edge_index = gw_wells_self_loops

    # Ensure 'pfas_sites' is a destination node type
    assert ('pfas_sites', 'distance', 'gw_wells') in data.edge_index_dict.keys(), "There is no edge from 'pfas_sites' to 'gw_wells'"
    assert ('gw_wells', 'distance', 'pfas_sites') in data.edge_index_dict.keys(), "There is no edge from 'gw_wells' to 'pfas_sites'"
    assert ('pfas_sites', 'self_loop', 'pfas_sites') in data.edge_index_dict.keys(), "There is no self-loop edge for 'pfas_sites'"
    assert ('gw_wells', 'self_loop', 'gw_wells') in data.edge_index_dict.keys(), "There is no self-loop edge for 'gw_wells'"

    logger.info(f"Edges from 'pfas_sites' to 'gw_wells': {data['pfas_sites', 'distance', 'gw_wells'].edge_index.shape[1]}")
    logger.info(f"Edges from 'gw_wells' to 'pfas_sites': {data['gw_wells', 'distance', 'pfas_sites'].edge_index.shape[1]}")
    logger.info(f"Self-loop edges for 'pfas_sites': {data['pfas_sites', 'self_loop', 'pfas_sites'].edge_index.shape[1]}")
    logger.info(f"Self-loop edges for 'gw_wells': {data['gw_wells', 'self_loop', 'gw_wells'].edge_index.shape[1]}")

    return data