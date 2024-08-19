import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import random
import itertools
import concurrent.futures
import uuid
from joblib import Parallel, delayed
from libs.plot_funs import plot_sum_pfas, plot_pred_sum_pfas, plot_loss_curve, plot_predictions
from libs.GNN_models import get_model_by_name
#from libs.hetero_data_creation import MainGNNModel
from libs.utils import cleanup_models, remove_torch_geometry_garbage, remove_predictions, logging_fitting_results, get_features_string, setup_logging, save_predictions
from libs.load_data import load_dataset, load_pfas_gw
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import HeteroConv
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
## set cuda device




def train(model, data, optimizer, criterion, device, logger, epochs=100, patience=25, args=None):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    ### generate a random number for the best_model.pth
    ## assert both gw_wells and pfas_sites are in the data
    assert 'sw_stations' in data.x_dict.keys(), "sw_stations is missing in data"
    assert 'gw_wells' in data.x_dict.keys(), "gw_wells is missing in data"
    assert 'pfas_sites' in data.x_dict.keys(), "pfas_sites is missing in data"
    serial_number = uuid.uuid4().hex
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        pred = out['gw_wells'][data['gw_wells'].train_mask]
        target = data['gw_wells'].x[data['gw_wells'].train_mask, 0].unsqueeze(1).to(device)
        gw_loss = criterion(pred, target)
        ## assert gw_loss type 
        assert isinstance(gw_loss, torch.Tensor), "gw_loss is not a tensor"
        pred = out['sw_stations'][data['sw_stations'].train_mask]
        target = data['sw_stations'].x[data['sw_stations'].train_mask, 0].unsqueeze(1).to(device)
        sw_loss = criterion(pred, target)

        loss = gw_loss + sw_loss

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = out['gw_wells'][data['gw_wells'].val_mask]
            val_target = data['gw_wells'].x[data['gw_wells'].val_mask, 0].unsqueeze(1).to(device)
            gw_val_loss = criterion(val_pred, val_target)

            val_pred = out['sw_stations'][data['sw_stations'].val_mask]
            val_target = data['sw_stations'].x[data['sw_stations'].val_mask, 0].unsqueeze(1).to(device)
            sw_val_loss = criterion(val_pred, val_target)

            val_loss = gw_val_loss + sw_val_loss

            val_losses.append(val_loss.item())

        model.train()
        if args.get("verbose", False) and (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Training Loss: {loss.item():.2f}, Validation Loss: {val_loss.item():.2f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), f'models/best_model_{serial_number}.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f'Early stopping at epoch {epoch+1}')
            break
    if args.get("plot", False):
        plot_loss_curve(train_losses, val_losses, logger)
    return train_losses, val_losses, serial_number


def evaluate(pfas_gw, model, data, criterion, device, serial_number,node_name, args=None, logger=None):

    assert 'geometry' in pfas_gw.columns, "geometry column is missing in pfas_gw"
    model.load_state_dict(torch.load(f'models/best_model_{serial_number}.pth'))
    model.eval()

    with torch.no_grad():
        train_pred = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)[node_name][data[node_name].train_mask]
        train_target = data[node_name].x[data[node_name].train_mask, 0].unsqueeze(1).to(device)
        train_loss = criterion(train_pred, train_target).item()

        val_pred = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)[node_name][data[node_name].val_mask]
        val_target = data[node_name].x[data[node_name].val_mask, 0].unsqueeze(1).to(device)
        val_loss = criterion(val_pred, val_target).item()

        test_pred = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)[node_name][data[node_name].test_mask]
        test_target = data[node_name].x[data[node_name].test_mask, 0].unsqueeze(1).to(device)
        test_loss = criterion(test_pred, test_target).item()

        ### prediction for unsampled data
        if node_name == 'gw_wells':
            unsampled_pred = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)[node_name][data[node_name].unsampled_mask]
            logger.info(f"Number of unsampled data: {len(unsampled_pred)} and {unsampled_pred}")
            unsampled_target = data[node_name].x[data[node_name].unsampled_mask, 0].unsqueeze(1).to(device)
        else:
            unsampled_pred = None
            unsampled_target = None
        save_predictions(pfas_gw, train_pred, val_pred, test_pred, data, unsampled_pred, device, args, serial_number, node_name)

        if args.get("verbose", False):
            logging_fitting_results(train_loss, val_loss, test_loss, train_target, train_pred, val_target, val_pred, test_target, test_pred,serial_number, logger)

    if args.get("plot", False):
        if unsampled_pred is not None:
            plot_predictions(train_target, train_pred, val_target, val_pred, test_target, test_pred, unsampled_pred, unsampled_target, logger, node_name)
        else:
            plot_predictions(train_target, train_pred, val_target, val_pred, test_target, test_pred, None, None, logger, node_name)

    return train_loss, val_loss, test_loss


def train_and_evaluate(device, data, pfas_gw,pfas_sw, in_channels_dict, edge_attr_dict, logger, out_channels, epochs, lr, weight_decay, args):
    ### assert both gw_wells and pfas_sites are in the in_channels_dict
    assert 'gw_wells' in in_channels_dict.keys(), "gw_wells is missing in in_channels_dict"
    assert 'pfas_sites' in in_channels_dict.keys(), "pfas_sites is missing in in_channels_dict"
    assert 'sw_stations' in in_channels_dict.keys(), "sw_stations is missing in in_channels_dict"
    
    ## assert sw_stations in data 
    assert "gw_wells" in data.x_dict.keys(), "gw_wells is missing in data"
    assert 'pfas_sites' in data.x_dict.keys(), "pfas_sites is missing in data"
    assert 'sw_stations' in data.x_dict.keys(), "sw_stations is missing in data"
    

    model = get_model_by_name(args['gnn_model'], in_channels_dict = in_channels_dict, edge_attr_dict = edge_attr_dict, out_channels=out_channels, aggregation=args['aggregation']).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    criterion = nn.MSELoss()
    train_losses, val_losses, serial_number = train(model, data, optimizer, criterion, device, logger, epochs=epochs, args=args)
    gw_train_loss, gw_val_loss, gw_test_loss = evaluate(pfas_gw, model, data, criterion, device, serial_number, node_name='gw_wells', args=args, logger=logger)
    sw_train_loss, sw_val_loss, sw_test_loss = evaluate(pfas_sw, model, data, criterion, device, serial_number, node_name='sw_stations', args=args, logger=logger)
    logger.info(f"Train loss: {sw_train_loss:.2f}, Validation loss: {sw_val_loss:.2f}, Test loss: {sw_test_loss:.2f}")
    logger.info(f"Train loss: {gw_train_loss:.2f}, Validation loss: {gw_val_loss:.2f}, Test loss: {gw_test_loss:.2f}")
    return (gw_train_loss, gw_val_loss, gw_test_loss)


def get_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = random.choice(['0', '1'])
    ### check if device usage is less than 80%
    device = torch.device('cuda' if torch.cuda.is_available() else Exception("No GPU available"))
    ### check
    return device


def generate_data_train_and_evaluate(out_channels, epochs, lr, weight_decay, distance, args, logger, single_none_parallel_run):

    device = get_device()
    in_channels_dict = {
        'pfas_sites': len(args['gw_features']) + 2,
        'gw_wells': len(args['gw_features']) + 2,
        'sw_stations': len(args['gw_features']) + 2
    }
    data, pfas_gw, pfas_sw = load_dataset(args, device, logger)
    edge_attr_dict =  data.edge_attr_dict
    if args.get("verbose", False):
        print("=========================================")
        print(f"#################### {data} ####################")
        print("=========================================")
    time.sleep(1)
    def single_iteration(_):
        return train_and_evaluate(device, data, pfas_gw,pfas_sw, in_channels_dict,edge_attr_dict, logger, out_channels=out_channels, epochs=epochs, lr=lr, weight_decay=weight_decay, args=args)

    if not single_none_parallel_run:
        # Run the iterations in parallel using ThreadPoolExecutor
        max_workers = args.get("repeated_k_fold", 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            best_losses = list(executor.map(single_iteration, range(max_workers)))

        return best_losses
    else:
        return single_iteration(0)


def wrapped_experiment(params):
    out_channels, epochs, lr, weight_decay, distance, gw_features,  gnn_model, aggregation = params
    return experiment(
        out_channels,
        epochs,
        lr,
        weight_decay,
        distance,
        gw_features,
        gnn_model,
        aggregation,

        single_none_parallel_run=False,
    )

def main(single_none_parallel_run=False):

    # Define hyperparameter grid
    out_channels_options = [16, 32, 48, 64]
    epochs_options = [500, 1000]
    lr_options = [0.001, 0.005, 0.01]
    weight_decay_options = [0.001, 0.005, 0.01]
    distance_options = [5000, 7500, 10000]
    gw_features_options = [['kriging_output_SWL_250m', 'DEM_250m'], ['DEM_250m'], []]
    gnn_models = ['leaky_relu_attention', 'leaky_relu', 'parametric_relu', 'tanh', 'relu', 'simple_GNNModel', 'ComplexGNNModel']
    aggregations = ['mean', 'add', 'max']
    # Generate all combinations of hyperparameters
    all_combinations = list(itertools.product(
        out_channels_options,
        epochs_options,
        lr_options,
        weight_decay_options,
        distance_options,
        gw_features_options,
        gnn_models,
        aggregations
    ))
    print(f"number of all combinations: {len(all_combinations)}")

    if single_none_parallel_run:
        single_experiment_execution()
    else:
        parallel_experiments_execution(all_combinations)


def single_experiment_execution():
    lr_options = 0.001
    epochs_options = 1000
    out_channels_options = 64
    weight_decay_options = 0.001
    distance_options = 7500
    #"geomorphons_250m_250Dis",  "LC22_EVH_220_250m", "MI_geol_poly_250m"
    gw_features_options = ['DEM_250m','kriging_output_SWL_250m','Aquifer_Characteristics_Of_Glacial_Drift_250m'] #'Aquifer_Characteristics_Of_Glacial_Drift_250m']#,'kriging_output_SWL_250m' ,'MI_geol_poly_250m','Aquifer_Characteristics_Of_Glacial_Drift_250m']#,'MI_geol_poly_250m', 'DEM_250m', 'kriging_output_SWL_250m']#, 'landforms_250m_250Dis', 'geomorphons_250m_250Dis', 'LC22_EVH_220_250m', 'Aquifer_Characteristics_Of_Glacial_Drift_250m']#, 'landforms_250m_250Dis', 'MI_geol_poly_250m']
    #,'Aquifer_Characteristics_Of_Glacial_Drift_250m']#, "landforms_250m_250Dis",  'MI_geol_poly_250m']
    # 'enhanced_sage_conv', 'prelu_attention' 'leaky_relu_attention', 'leaky_relu', 'parametric_relu', 'tanh', 'relu', 'simple_GNNModel', 'ComplexGNNModel'
    gnn_model = 'prelu' #'prelu_attention'
    aggregation = 'sum'
    all_combinations = [(out_channels_options, epochs_options, lr_options, weight_decay_options, distance_options, gw_features_options, gnn_model, aggregation)]
    ## choose random combination
    params = random.choice(all_combinations)
    experiment(*params, single_none_parallel_run = True)


def parallel_experiments_execution(all_combinations):
    # Use parallel processing to run experiments
    all_dataframes = Parallel(n_jobs=50)(delayed(wrapped_experiment)(params) for params in all_combinations)
    save_results(all_dataframes)

def save_results(all_dataframes):
    all_dataframes = pd.concat(all_dataframes)
    all_dataframes.to_csv('results/GridSearchResults.csv', index=False)
    print("All dataframes")
    print(all_dataframes)



def experiment(out_channels, epochs, lr, weight_decay, distance, gw_features, gnn_model, aggregation,  single_none_parallel_run=True):
    logger = setup_logging()
    args = {
        "repeated_k_fold": 10,
        "verbose": single_none_parallel_run,
        "plot": single_none_parallel_run,
        "data_dir": "/data/MyDataBase/HuronRiverPFAS/",
        "pfas_gw_columns": ['sum_PFAS'],
        "pfas_sites_columns": ['Industry'],
        "pfas_sw_station_columns": ['sum_PFAS'],
        "gw_features": gw_features,
        "distance_threshold": distance,
        'gnn_model': gnn_model,
        'aggregation': aggregation
    }

    best_losse = generate_data_train_and_evaluate(out_channels, epochs, lr, weight_decay, distance, args, logger, single_none_parallel_run)
    print(f"Best losses: {best_losse}")

    if single_none_parallel_run:
        return best_losse
    # Create DataFrame with correct columns
    df = pd.DataFrame(best_losse, columns=['train_loss', 'val_loss', 'test_loss'])
    df = df.median()
    df = df.to_frame().T
    df['out_channels'] = out_channels
    df['epochs'] = epochs
    df['lr'] = lr
    df['weight_decay'] = weight_decay
    df['distance'] = distance
    df['gw_features'] = get_features_string(gw_features)
    df['gnn_model'] = gnn_model
    df['aggregation'] = aggregation

    return df

def cleanup_gpu_memory():
    torch.cuda.empty_cache()


if __name__ == "__main__":
    #os.makedirs('results', exist_ok=True)
    #cleanup_gpu_memory()
    #cleanup_models()
    #remove_torch_geometry_garbage()
    #remove_predictions()
    main(single_none_parallel_run = True)
    #remove_torch_geometry_garbage()
    print("Done")