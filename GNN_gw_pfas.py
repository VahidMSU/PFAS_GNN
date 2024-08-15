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
from libs.plot_funs import plot_sum_pfas, plot_pred_sum_pfas, plot_loss_curve, plot_predictions, plot_loss_histograms
from libs.GNN_models import MainGNNModel as GNNModel
from libs.utils import cleanup_models, remove_torch_geometry_garbage, remove_predictions, logging_fitting_results, get_features_string, setup_logging
from libs.load_data import load_dataset, load_pfas_gw
import matplotlib.pyplot as plt
## set cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
    
def uncertainty_analysis():
    logger = setup_logging("uncertainty_analysis.txt")
    logger.info("Starting uncertainty analysis")
    path = 'predictions_results/'
    files = os.listdir(path)
    ## ending with csv
    files = [file for file in files if file.endswith('.csv')]
    ## read all files
    dfs = [pd.read_csv(path + file) for file in files]
    print("Number of files: ", len(dfs))
    ## concatenate all dataframes
    all_data = pd.concat(dfs)
    ## group by WSSN
    grouped = all_data.groupby('WSSN')
    ## calculate mean and standard deviation
    mean = grouped.mean().reset_index()
    std = grouped.std().reset_index()
    
    ## merge mean and std
    mean_std = mean.merge(std, on='WSSN', suffixes=('_mean', '_std'))

    ## save mean_std
    mean_std[['WSSN', 'sum_PFAS_mean', 'pred_sum_PFAS_mean', 'pred_sum_PFAS_std']].rename(columns={'sum_PFAS_mean': 'obs_PFAS'}).to_csv('results/gw_wells_PFAS_predictions_mean_std.csv', index=False)
    plot_uncertainty_mean_vs_observed_PFAS('results/gw_wells_PFAS_predictions_mean_std.csv', logger)


def plot_uncertainty_mean_vs_observed_PFAS(path, logger):
    path = "results/gw_wells_PFAS_predictions_mean_std.csv"
    df = pd.read_csv(path)  # contains pred_sum_PFAS_mean, obs_PFAS, pred_sum_PFAS_std

    # Read PFAS data
    data_dir = "/data/MyDataBase/HuronRiverPFAS/"
    gw_features = []
    pfas_gw = load_pfas_gw(data_dir, gw_features, logger)   
    pfas_gw = pfas_gw[['WSSN', 'geometry']].merge(df, on='WSSN')

    # Sort the data by observed PFAS for a more meaningful plot
    logger.info(f"Number of samples before removing zero values: {len(pfas_gw)}")
    pfas_gw = pfas_gw.sort_values('obs_PFAS').reset_index(drop=True)
    pfas_gw = pfas_gw[pfas_gw['obs_PFAS'] > 0]  # Remove zero values
    logger.info(f"Number of samples after removing zero values: {len(pfas_gw)}")
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    # Plot the bars with error bars
    ax.bar(pfas_gw.index, pfas_gw['pred_sum_PFAS_mean'], yerr=pfas_gw['pred_sum_PFAS_std'], 
        capsize=4, color='lightblue', label='Predicted Mean ± Std')

    # Plot the observed PFAS values as points
    ax.plot(pfas_gw.index, pfas_gw['obs_PFAS'], 'ro', label='Observed PFAS')

    # Connect the observed PFAS values with lines
    ax.plot(pfas_gw.index, pfas_gw['obs_PFAS'], 'r-', alpha=0.7)

    # Set labels and title
    ax.set_xlabel('Groundwater Wells')
    ax.set_ylabel('Sum of PFAS (ng/L)')
    ax.set_title('Predicted Mean ± Std vs None-Zero Observed Sum of PFAS')

    # Add grid
    ax.grid(True)

    # Add legend
    ax.legend()

    # Save figure
    plt.savefig('figs/uncertainty_mean_vs_observed_PFAS_with_error_bars.png', dpi=300)
    plt.close()



def train(model, data, optimizer, criterion, device, logger, epochs=100, patience=25, args=None):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    ### generate a random number for the best_model.pth

    serial_number = uuid.uuid4().hex
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        pred = out['gw_wells'][data['gw_wells'].train_mask]
        target = data['gw_wells'].x[data['gw_wells'].train_mask, 0].unsqueeze(1).to(device)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_pred = out['gw_wells'][data['gw_wells'].val_mask]
            val_target = data['gw_wells'].x[data['gw_wells'].val_mask, 0].unsqueeze(1).to(device)
            val_loss = criterion(val_pred, val_target)
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


def save_predictions(pfas_gw, train_pred, val_pred, test_pred, data, unsampled_pred, device, args, serial_number):
    
    # Get node indices for train, validation, and test samples
    train_gw_node_index = data['gw_wells'].x[data['gw_wells'].train_mask, 1].to(device)
    val_gw_node_index = data['gw_wells'].x[data['gw_wells'].val_mask, 1].to(device)
    test_gw_node_index = data['gw_wells'].x[data['gw_wells'].test_mask, 1].to(device)
    unsampled_node_index = data['gw_wells'].x[data['gw_wells'].unsampled_mask, 1].to(device)
    
    # Concatenate predictions and WSSN values
    all_pred = torch.cat([train_pred, val_pred, test_pred, unsampled_pred], dim=0)
    all_wssn = torch.cat([train_gw_node_index, val_gw_node_index, test_gw_node_index, unsampled_node_index], dim=0)
    
    # Convert to DataFrame
    all_pred_wssn = pd.DataFrame({
        'gw_node_index': all_wssn.cpu().numpy().astype(int),  # Ensure WSSN is of type string
        'pred_sum_PFAS': all_pred.cpu().numpy().flatten()
    })
    
    #all_pred_wssn.to_csv('all_pred_gw_node_index.csv')
    pfas_gw = pfas_gw.merge(all_pred_wssn, on='gw_node_index', how='left')
    ## save pfas_gw
    pfas_gw[['WSSN','sum_PFAS', 'pred_sum_PFAS']].to_csv(f'predictions_results/pfas_gw_pred_{serial_number}.csv', index=False, float_format='%.4f')
    if args.get("plot", False): 
        plot_pred_sum_pfas(pfas_gw)
        plot_sum_pfas(pfas_gw)

def evaluate(pfas_gw, model, data, criterion, device, serial_number, args=None, logger=None):

    assert 'geometry' in pfas_gw.columns, "geometry column is missing in pfas_gw"
    model.load_state_dict(torch.load(f'models/best_model_{serial_number}.pth'))
    model.eval()

    with torch.no_grad():
        train_pred = model(data.x_dict, data.edge_index_dict)['gw_wells'][data['gw_wells'].train_mask]
        train_target = data['gw_wells'].x[data['gw_wells'].train_mask, 0].unsqueeze(1).to(device)
        train_loss = criterion(train_pred, train_target).item()

        val_pred = model(data.x_dict, data.edge_index_dict)['gw_wells'][data['gw_wells'].val_mask]
        val_target = data['gw_wells'].x[data['gw_wells'].val_mask, 0].unsqueeze(1).to(device)
        val_loss = criterion(val_pred, val_target).item()

        test_pred = model(data.x_dict, data.edge_index_dict)['gw_wells'][data['gw_wells'].test_mask]
        test_target = data['gw_wells'].x[data['gw_wells'].test_mask, 0].unsqueeze(1).to(device)
        test_loss = criterion(test_pred, test_target).item()

        ### prediction for unsampled data
        unsampled_pred = model(data.x_dict, data.edge_index_dict)['gw_wells'][data['gw_wells'].unsampled_mask]
        logger.info(f"Number of unsampled data: {len(unsampled_pred)} and {unsampled_pred}")
        unsampled_target = data['gw_wells'].x[data['gw_wells'].unsampled_mask, 0].unsqueeze(1).to(device)

        save_predictions(pfas_gw, train_pred, val_pred, test_pred, data, unsampled_pred, device, args, serial_number)

        if args.get("verbose", False):
            logging_fitting_results(train_loss, val_loss, test_loss, train_target, train_pred, val_target, val_pred, test_target, test_pred,serial_number, logger)

    if args.get("plot", False):
        plot_predictions(train_target, train_pred, val_target, val_pred, test_target, test_pred, unsampled_pred, unsampled_target)
        
    return train_loss, val_loss, test_loss


def train_and_evaluate(device, data, pfas_gw, in_channels_dict, logger, out_channels=48, epochs=500, lr=0.01, weight_decay=0.01, args=None):
    try:
        model = GNNModel(in_channels_dict, out_channels=out_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        train_losses, val_losses, serial_number = train(model, data, optimizer, criterion, device, logger, epochs=epochs, args=args) 
        train_loss, val_loss, test_loss = evaluate(pfas_gw, model, data, criterion, device, serial_number, args=args, logger=logger)

        logger.info(f"Train loss: {train_loss:.2f}, Validation loss: {val_loss:.2f}, Test loss: {test_loss:.2f}")
        return (train_loss, val_loss, test_loss)
    except Exception as e:
        logger.error(f"Error in training and evaluation: {e}")
        return (np.nan, np.nan, np.nan)

def get_device():
    ### check if device usage is less than 80%
    device = torch.device('cuda' if torch.cuda.is_available() else Exception("No GPU available"))
    ### check
    return device
    

def generate_data_train_and_evaluate(out_channels, epochs, lr, weight_decay, distance, args, logger, single_none_parallel_run):
    
    device = get_device()
    in_channels_dict = {
        'pfas_sites': len(args['gw_features']) + 2,
        'gw_wells': len(args['gw_features']) + 2,
    }
    data, pfas_gw = load_dataset(args, device, logger)
    if args.get("verbose", False):
        print("=========================================")
        print(f"#################### {data} ####################")
        print("=========================================")
    time.sleep(5)
    def single_iteration(_):
        return train_and_evaluate(device, data, pfas_gw, in_channels_dict,logger, out_channels=out_channels, epochs=epochs, lr=lr, weight_decay=weight_decay, args=args)
    
    if not single_none_parallel_run:
        # Run the iterations in parallel using ThreadPoolExecutor
        max_workers = args.get("repeated_k_fold", 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            best_losses = list(executor.map(single_iteration, range(max_workers)))
        
        return best_losses
    else:
        return single_iteration(0)


def wrapped_experiment(params):
    out_channels, epochs, lr, weight_decay, distance, gw_features = params
    return experiment(
        out_channels,
        epochs,
        lr,
        weight_decay,
        distance,
        gw_features,
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

    # Generate all combinations of hyperparameters
    all_combinations = list(itertools.product(
        out_channels_options,
        epochs_options,
        lr_options,
        weight_decay_options,
        distance_options,
        gw_features_options
    ))
    print(f"number of all combinations: {len(all_combinations)}")
    
    if single_none_parallel_run:
        single_experiment_execution()
    else:
        parallel_experiments_execution(all_combinations)


def single_experiment_execution():
    lr_options = 0.001
    epochs_options = 1000
    out_channels_options = 16
    weight_decay_options = 0.005
    distance_options = 10000
    gw_features_options = []
    all_combinations = [(out_channels_options, epochs_options, lr_options, weight_decay_options, distance_options, gw_features_options)]
    ## choose random combination
    params = random.choice(all_combinations)
    experiment(*params, single_none_parallel_run = True)


def parallel_experiments_execution(all_combinations):
    # Use parallel processing to run experiments
    all_dataframes = Parallel(n_jobs=50)(delayed(wrapped_experiment)(params) for params in all_combinations)
    all_dataframes = pd.concat(all_dataframes)
    all_dataframes.to_csv('results/GridSearchResults.csv', index=False)
    print("All dataframes")
    print(all_dataframes)



def experiment(out_channels, epochs, lr, weight_decay, distance, gw_features, single_none_parallel_run=True):
    logger = setup_logging()
    args = {
        "repeated_k_fold": 10,
        "verbose": single_none_parallel_run,
        "plot": single_none_parallel_run,
        "data_dir": "/data/MyDataBase/HuronRiverPFAS/",
        "pfas_gw_columns": ['sum_PFAS'],
        "pfas_sites_columns": ['Industry'],
        "gw_features": gw_features,
        "distance_threshold": distance
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
    
    return df
def cleanup_gpu_memory():
    torch.cuda.empty_cache()





if __name__ == "__main__":
   # os.makedirs('results', exist_ok=True)
   # cleanup_gpu_memory()
   # cleanup_models()
   # remove_torch_geometry_garbage()
   # remove_predictions()
   # main(single_none_parallel_run = False)
    uncertainty_analysis()
    plot_loss_histograms()
    remove_torch_geometry_garbage()
    print("Done")