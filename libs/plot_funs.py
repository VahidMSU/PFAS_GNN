import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 


def plot_distribution(pfas_gw, logger, name='gw'):
    logger.info("stage:plot_distribution ==##== Plotting sum of PFAS distribution")
    plt.figure()
    pfas_gw['sum_PFAS'].plot.hist(bins=100)
    plt.xlabel('Sum of PFAS (ng/L)')
    plt.ylabel('Frequency')
    plt.yscale("symlog")
    plt.title('Distribution of Sum of PFAS')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    os.makedirs('figs', exist_ok=True)
    plt.savefig(f'figs/sum_pfas_{name}_distribution.png', dpi=300)
    plt.close()

def plot_site_samples(train_gw, val_gw, test_gw, pfas_sites, logger, name='gw'):
    logger.info("stage:plot_site_samples ==##== Plotting sampled sites")
    ax, fig = plt.subplots(figsize=(8, 8))
    huron_bounds = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl").to_crs("EPSG:4326")
    huron_bounds.boundary.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=1)
    pfas_sites[['Industry','geometry']].to_crs("EPSG:4326").plot(ax=plt.gca(), color='lightgray', edgecolor='black', linewidth=0.5, marker='^')

    train_gw[['geometry']].to_crs("EPSG:4326").plot(ax=plt.gca(), color='blue', edgecolor='black', linewidth=0.5)
    val_gw[['geometry']].to_crs("EPSG:4326").plot(ax=plt.gca(), color='green', edgecolor='black', linewidth=0.5)
    test_gw[['geometry']].to_crs("EPSG:4326").plot(ax=plt.gca(), color='red', edgecolor='black', linewidth=0.5)
    number_of_sites = len(pfas_sites)
    number_of_train = len(train_gw)
    number_of_val = len(val_gw)
    number_of_test = len(test_gw)
    ## annotate
    plt.annotate(f"Number of sites: {number_of_sites}\nNumber of train: {number_of_train}\nNumber of val: {number_of_val}\nNumber of test: {number_of_test}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Splitted PFAS samples in Huron River Basin')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.legend(["Huron River Basin", 'PFAS sites', 'Train', 'Validation', 'Test'], loc = 'lower left')
    plt.tight_layout()
    plt.savefig(f'figs/{name}_sampled_sites.png', dpi=300)

    plt.close()

def plot_sum_pfas(pfas_gw, node_name):
    assert "geometry" in pfas_gw.columns, "geometry column is missing in pfas_gw"
    assert len(pfas_gw) > 0, "pfas_gw is empty"
    assert "sum_PFAS" in pfas_gw.columns, "sum_PFAS column is missing in pfas_gw"
    ## assert no negative values in sum_PFAS
    assert pfas_gw['sum_PFAS'].min() >= 0,"There are negative values in sum_PFAS"
    bounds = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl").to_crs("EPSG:4326")

    ### verify all classes are present
   # assert all(
   #     i in pfas_gw['sum_PFAS_class'].unique() for i in [0, 1, 2]
    #), "There are missing classes in sum_PFAS_class"
    colors = ['green', 'blue', 'red' , 'grey']
    pfas_gw = gpd.GeoDataFrame(pfas_gw, geometry='geometry', crs='EPSG:26990').to_crs("EPSG:4326")

    plt.figure(figsize=(8, 8))
    bounds.boundary.plot(ax=plt.gca(), facecolor='none', edgecolor='black', linewidth=1)
    if node_name == 'sw_stations':
        legend_labels = ['0-200', '200-500', f'500-{pfas_gw["sum_PFAS"].max():.1f}', 'unknown']
    else:
        legend_labels = ['0-1','1-10',f'10-{pfas_gw["sum_PFAS"].max():.1f}', 'unknown']
    ### sort before plotting to show higher values on top
    #pfas_gw = pfas_gw.sort_values('sum_PFAS', ascending=False)
    ## sort in a way that we first see red, then blue, then green, then gray
    pfas_gw = pfas_gw.sort_values('sum_PFAS_class')
    for i, (name, group) in enumerate(pfas_gw.groupby('sum_PFAS_class', observed=True)):
        if colors[i] == 'grey':
            group.plot(ax=plt.gca(), facecolor='none', edgecolor=colors[i], linewidth=0.5, alpha=0.8)
        else:
            group.plot(ax=plt.gca(), color=colors[i],alpha=0.8)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(["Huron River Basin"] + legend_labels, loc='lower left')
    if node_name == 'gw_wells':
        plt.title(f'#{pfas_gw["WSSN"].nunique()} water wells with unique WSSN')
    elif node_name == 'sw_stations':
        plt.title(f'#{pfas_gw["SiteCode"].nunique()} sites with unique SiteCode')
    else:
        plt.title('#ERROR IN TITLE')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'figs/sum_{node_name}_PFAS.png', dpi=300)
    plt.close()



def plot_pred_sum_pfas(pfas_gw, node_name):
    assert "geometry" in pfas_gw.columns, "geometry column is missing in pfas_gw"
    assert len(pfas_gw) > 0, "pfas_gw is empty"
    assert "pred_sum_PFAS" in pfas_gw.columns, "pred_sum_PFAS column is missing in pfas_gw"
    
    # Load the boundaries
    bounds = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl").to_crs("EPSG:4326")
    
    # Replace negative values with 0
    pfas_gw['pred_sum_PFAS'] = pfas_gw['pred_sum_PFAS'].clip(lower=0)
    
    # Define color and label mappings based on node_name
    if node_name == 'gw_wells':
        class_bins = [-1, 1, 10, 50, 200]
        class_colors = {0: 'green', 1: 'blue', 2: 'red'}
        class_labels = {0: '0-1', 1: '1-10', 2: '10-200'}
    else:
        class_bins = [-1, 0, 200, 500, 2000]
        class_colors = {0: 'green', 1: 'yellow', 2: 'blue', 3: 'red'}
        class_labels = {0: '0', 1: '0-200', 2: '200-500', 3: '500-2000'}
    
    # Classify the data
    pfas_gw['pred_sum_PFAS_class'] = pd.cut(pfas_gw['pred_sum_PFAS'], bins=class_bins, labels=range(len(class_bins)-1))
    
    # Convert to GeoDataFrame and ensure correct CRS
    pfas_gw = gpd.GeoDataFrame(pfas_gw, geometry='geometry', crs='EPSG:26990').to_crs("EPSG:4326")
    
    # Plot setup
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Plot the boundaries
    bounds.boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    
    # Sort values for better visualization
    pfas_gw = pfas_gw.sort_values('pred_sum_PFAS')
    
    # Plot each class with its assigned color
    for class_value, color in class_colors.items():
        class_group = pfas_gw[pfas_gw['pred_sum_PFAS_class'] == class_value]
        if not class_group.empty:
            class_group.plot(ax=ax, color=color, edgecolor='black', linewidth=0.5, alpha=0.5, label=class_labels[class_value])
    
    # Set axis labels
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Set axis ticks and format
    plt.xticks(np.linspace(pfas_gw.total_bounds[0], pfas_gw.total_bounds[2], 4))
    plt.yticks(np.linspace(pfas_gw.total_bounds[1], pfas_gw.total_bounds[3], 4))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
    # Add legend and title
    plt.legend(loc='lower left')
    plt.title(f'Predicted Sum of PFAS for {len(pfas_gw)} samples with range: {pfas_gw["pred_sum_PFAS"].min():.2f} - {pfas_gw["pred_sum_PFAS"].max():.2f}')
    
    # Add grid
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    
    # Save the plot
    plt.savefig(f'figs/pred_{node_name}_sum_PFAS.png', dpi=300)
    plt.close()

    
def plot_predictions(train_target, train_pred, val_target, val_pred, test_target, test_pred, unsampled_pred, unsampled_target, logger, node_name):
    logger.info("###################################################")
    logger.info(f"Number of train samples: %d {len(train_target)} with range: {train_target.min():.2f} - {train_target.max():.2f}")
    logger.info(f"Number of val samples: %d {len(val_target)} with range: {val_target.min():.2f} - {val_target.max():.2f}")
    logger.info(f"Number of test samples: %d {len(test_target)} with range: {test_target.min():.2f} - {test_target.max():.2f}")
    if unsampled_pred is not None:
        logger.info(f"Number of unsampled samples: %d {len(unsampled_target)} with range: {unsampled_target.min():.2f} - {unsampled_target.max():.2f}")
    logger.info("###################################################")

    plt.figure()
    plt.scatter(train_target.cpu().numpy(), train_pred.cpu().numpy(), label='Training', alpha=0.5)
    plt.scatter(val_target.cpu().numpy(), val_pred.cpu().numpy(), label='Validation', alpha=0.5)
    plt.scatter(test_target.cpu().numpy(), test_pred.cpu().numpy(), label='Test', alpha=0.5)
    if unsampled_pred is not None:
        #plt.scatter(unsampled_target.cpu().numpy(), unsampled_pred.cpu().numpy(), label='Unsampled', alpha=0.5)
    
        ## save unsampled predictions in a text file to inspect
        unsampled_df = pd.DataFrame({
            'target': unsampled_target.cpu().numpy().flatten(),
            'pred': unsampled_pred.cpu().numpy().flatten()
        })
    #unsampled_df.to_csv('unsampled_predictions.csv', index=False)

    plt.title(f'Predicted vs True Sum of PFAS for\n{len(test_target)+len(val_target)+len(train_target)} water wells with unique {node_name}')
    plt.xlabel('True Sum of PFAS (ng/L)')
    plt.ylabel('Predicted Sum of PFAS (ng/L)')
    plt.legend()
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.xscale('log')

    plt.savefig(f'figs/{node_name}_predictions.png', dpi=300)
    plt.close()


def plot_histogram(all_dataframes):
    plt.figure()
    all_dataframes['train_loss'].plot.hist(alpha=0.5, bins=50, label='train_loss')
    all_dataframes['val_loss'].plot.hist(alpha=0.5, bins=50, label='val_loss')
    all_dataframes['test_loss'].plot.hist(alpha=0.5, bins=50, label='test_loss')
    plt.legend()
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.xscale('symlog')  ## other options: log, symlog, linear
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.savefig('figs/loss_histogram.png', dpi=300)
    plt.close()

def plot_loss_curve(train_losses, val_losses, logger):
    logger.info(f"stage:plot_loss_curve ==##== Number of epochs: {len(train_losses)}")
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('figs/loss_curve.png', dpi=300)
    plt.close()

