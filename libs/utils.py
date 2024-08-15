
import os
import glob
import logging
import pandas as pd
import numpy as np
import pandas as pd


def cleanup_temp_files():
    ## all csv files
    files = glob.glob('temp/*.csv')
    for f in files:
        os.remove(f)

def setup_logging(path='GNN_gw_pfas.txt', verbose=True):
    ## if verbose is False, do not write on the console or 
    if verbose:
        with open(path, 'w') as f:
            f.write('')
        # Configure logger to write to both console and file
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

        # Create a logger object
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(path)
        fh.setLevel(logging.INFO)

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        # define error logger
        error_logger = logging.getLogger('error')
        error_logger.setLevel(logging.ERROR)
        error_logger.addHandler(fh)
        error_logger.addHandler(ch)
        return logger
    else:
        with open(path, 'w') as f:
            f.write('')
        # Configure logger to write to both console and file
        logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')

        # Create a logger object
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(path)
        fh.setLevel(logging.ERROR)

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        # define error logger
        error_logger = logging.getLogger('error')
        error_logger.setLevel(logging.ERROR)
        error_logger.addHandler(fh)
        error_logger.addHandler(ch)
        return logger

def get_features_string(gw_features):
    if "DEM_250m" in gw_features and "kriging_output_SWL_250m" in gw_features:
        return "SWL and DEM"
    elif "DEM_250m" in gw_features:
        return "DEM"
    else:
        return "None"

def cleanup_models():
    for f in glob.glob('models/*.pth'):
        os.remove(f)
    

def remove_torch_geometry_garbage():
    ##get current directory
    current_dir = os.getcwd()
    ## remove all files with torch_geometric.nn
    files = glob.glob(f'{current_dir}/torch_geometric.nn*')
    for f in files:
        os.remove(f)
    files = glob.glob(f'{current_dir}/__pycache__/torch_geometric.nn*')
    for f in files:
        os.remove(f)
   
def remove_predictions():
    os.makedirs('predictions_results', exist_ok=True)
    files = glob.glob('predictions_results/pfas_gw_pred_*.csv')
    for f in files:
        os.remove(f)



def logging_fitting_results(train_loss, val_loss, test_loss, train_target, train_pred, val_target, val_pred, test_target, test_pred, serial_number, logger):
    logger.info(f"######Serial Number: {serial_number}######")
    logger.info(f"######Training Loss: {train_loss:.2f}######")
    logger.info(f"######Validation Loss: {val_loss:.2f}######")
    logger.info(f"######Test Loss: {test_loss:.2f}######")

    def report_zero_and_small_values(tensor):
        numpy_array = tensor.cpu().numpy()
        zero_values = np.sum(numpy_array == 0)
        small_values = np.sum(numpy_array < 0.01)
        return zero_values, small_values

    data = []

    for name, target, pred in [('train', train_target, train_pred), ('val', val_target, val_pred), ('test', test_target, test_pred)]:
        target_zeros, target_smalls = report_zero_and_small_values(target)
        pred_zeros, pred_smalls = report_zero_and_small_values(pred)

        data.append({
            'Set': name,
            'Target Shape': target.cpu().numpy().shape,
            'Target Range Min': target.cpu().numpy().min(),
            'Target Range Max': target.cpu().numpy().max(),
            'Target Zero Values': target_zeros,
            'Target Small Values <0.01': target_smalls,
            'Pred Shape': pred.cpu().numpy().shape,
            'Pred Range Min': pred.cpu().numpy().min(),
            'Pred Range Max': pred.cpu().numpy().max(),
            'Pred Zero Values': pred_zeros,
            'Pred Small Values <0.01': pred_smalls
        })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Print the DataFrame
    print(df)

    # Optionally, log the DataFrame (you can also save it to a file if needed)
    logger.info(f"\n{df}")

    return df
