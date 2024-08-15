import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 


def plot_distribution(pfas_gw, logger):
    logger.info("stage:plot_distribution ==##== Plotting sum of PFAS distribution")
    plt.figure()
    pfas_gw['sum_PFAS'].plot.hist(bins=100)
    plt.xlabel('Sum of PFAS (ng/L)')
    plt.ylabel('Frequency')
    plt.yscale("symlog")
    plt.title('Distribution of Sum of PFAS')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/sum_pfas_distribution.png', dpi=300)
    plt.close()

def plot_site_samples(train_gw, val_gw, test_gw, pfas_sites, logger):
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
    plt.savefig('figs/sampled_sites.png', dpi=300)

    plt.close()

def plot_sum_pfas(pfas_gw):
    assert "geometry" in pfas_gw.columns, "geometry column is missing in pfas_gw"
    assert len(pfas_gw) > 0, "pfas_gw is empty"
    assert "sum_PFAS" in pfas_gw.columns, "sum_PFAS column is missing in pfas_gw"
    ## assert no negative values in sum_PFAS
    assert pfas_gw['sum_PFAS'].min() >= 0,"There are negative values in sum_PFAS"


    ### verify all classes are present
    assert all(
        i in pfas_gw['sum_PFAS_class'].unique() for i in [0, 1, 2]
    ), "There are missing classes in sum_PFAS_class"
    colors = ['green', 'blue', 'red', 'gray']
    pfas_gw = gpd.GeoDataFrame(pfas_gw, geometry='geometry', crs='EPSG:26990').to_crs("EPSG:4326")
    
    plt.figure(figsize=(8, 8))
    ### sort before plotting to show higher values on top
    pfas_gw = pfas_gw.sort_values('sum_PFAS', ascending=False)
    for i, (name, group) in enumerate(pfas_gw.groupby('sum_PFAS_class', observed=True)):
        if i == 3:
            group.plot(ax=plt.gca(), color=colors[i], edgecolor='black', linewidth=0.5, alpha=0.5, markersize=15)
        else:
            group.plot(ax=plt.gca(), color=colors[i], edgecolor='black', linewidth=0.5, alpha=0.5)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(['0', '0-10', '10-1000', 'unknown'])
    plt.title(f'Sum of PFAS for {len(pfas_gw)} samples with range: {pfas_gw["sum_PFAS"].min():.2f} - {pfas_gw["sum_PFAS"].max():.2f}')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.savefig('figs/sum_PFAS.png', dpi=300)
    plt.close()

def plot_pred_sum_pfas(pfas_gw):
    assert "geometry" in pfas_gw.columns, "geometry column is missing in pfas_gw"
    assert len(pfas_gw) > 0, "pfas_gw is empty"
    assert "pred_sum_PFAS" in pfas_gw.columns, "pred_sum_PFAS column is missing in pfas_gw"
    ### classify predictions into negative, 0, 0-10, 10-1000. NOTE: the must be a specific group for only zeros
    ### replace negative pfas_gw['pred_sum_PFAS'] with 0
    pfas_gw['pred_sum_PFAS'] = pfas_gw['pred_sum_PFAS'].clip(lower=0)
    pfas_gw['pred_sum_PFAS_class'] = pd.cut(pfas_gw['pred_sum_PFAS'], bins=[-1, 0, 0.5 ,10, 10000], labels=[0, 1, 2,3])
    colors = ['green','yellow' ,'blue' ,'red']
    pfas_gw = gpd.GeoDataFrame(pfas_gw, geometry='geometry', crs='EPSG:26990').to_crs("EPSG:4326")
    plt.figure()
    ### sort before plotting to show higher values on top
    pfas_gw = pfas_gw.sort_values('pred_sum_PFAS')
    for i, (name, group) in enumerate(pfas_gw.groupby('pred_sum_PFAS_class', observed=True)):
        group.plot(ax=plt.gca(), color=colors[i], edgecolor='black', linewidth=0.5, alpha=0.5)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ## only 4 ticks for x and y axis
    plt.xticks(np.linspace(pfas_gw.total_bounds[0], pfas_gw.total_bounds[2], 4))
    plt.yticks(np.linspace(pfas_gw.total_bounds[1], pfas_gw.total_bounds[3], 4))
    ## 2 decimal places for x and y axis
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    plt.legend(['0','0-0.5 (model error)' ,'0.5-10', '10-1000'])
    plt.title(f'Predicted Sum of PFAS for {len(pfas_gw)} samples with range: {pfas_gw["pred_sum_PFAS"].min():.2f} - {pfas_gw["pred_sum_PFAS"].max():.2f}')
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.savefig('figs/pred_sum_PFAS.png', dpi=300)
    plt.close()
    
def plot_predictions(train_target, train_pred, val_target, val_pred, test_target, test_pred, unsampled_pred, unsampled_target):
    
    plt.figure()
    plt.scatter(train_target.cpu().numpy(), train_pred.cpu().numpy(), label='Training', alpha=0.5)
    plt.scatter(val_target.cpu().numpy(), val_pred.cpu().numpy(), label='Validation', alpha=0.5)
    plt.scatter(test_target.cpu().numpy(), test_pred.cpu().numpy(), label='Test', alpha=0.5)
    plt.scatter(unsampled_target.cpu().numpy(), unsampled_pred.cpu().numpy(), label='Unsampled', alpha=0.5)
    
    ## save unsampled predictions in a text file to inspect
    unsampled_df = pd.DataFrame({
        'target': unsampled_target.cpu().numpy().flatten(),
        'pred': unsampled_pred.cpu().numpy().flatten()
    })
    #unsampled_df.to_csv('unsampled_predictions.csv', index=False)

    plt.title(f'Predicted vs True Sum of PFAS for {len(test_target)+len(val_target)+len(train_target)} samples')
    plt.xlabel('True Sum of PFAS (ng/L)')
    plt.ylabel('Predicted Sum of PFAS (ng/L)')
    plt.legend()
    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(0.001, 300)
    plt.ylim(0.01, 300)
    plt.savefig('figs/predictions.png', dpi=300)
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



def plot_loss_histograms():
    path = "/home/rafieiva/MyDataBase/codes/PFAS_GNN/GridSearchResults.csv"
    df = pd.read_csv(path)

    # Define a function to remove outliers using the IQR method
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Remove outliers from the loss columns
    df_no_outliers = df.copy()
    df_no_outliers = remove_outliers(df_no_outliers, 'train_loss')
    df_no_outliers = remove_outliers(df_no_outliers, 'val_loss')
    df_no_outliers = remove_outliers(df_no_outliers, 'test_loss')

    # Plot the histograms without outliers
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    plt.hist(df_no_outliers['train_loss'], bins=30, color='skyblue')
    plt.title('Train Loss Distribution (No Outliers)')
    plt.xlabel('Train Loss')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(df_no_outliers['val_loss'], bins=30, color='lightgreen')
    plt.title('Validation Loss Distribution (No Outliers)')
    plt.xlabel('Validation Loss')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(df_no_outliers['test_loss'], bins=30, color='lightcoral')
    plt.title('Test Loss Distribution (No Outliers)')
    plt.xlabel('Test Loss')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'figs/loss_histograms_no_outliers.png', dpi=300)
    plt.close()
