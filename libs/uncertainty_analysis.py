from libs.load_data import load_pfas_gw, load_pfas_sites, load_dataset
from libs.utils import setup_logging
import os 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Patch





def uncertainty_analysis(logger):  

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
    _5th_percentile = grouped.quantile(0.05).reset_index()
    _95th_percentile = grouped.quantile(0.95).reset_index()
    ## merge mean and std
    mean_std = mean.merge(std, on='WSSN', suffixes=('_mean', '_std'))
    _95PPU_ = _95th_percentile.merge(_5th_percentile, on='WSSN', suffixes=('_95th', '_5th'))
    ## merge mean_std with _95PPU_
    mean_std = mean_std.merge(_95PPU_, on='WSSN')
    ## save mean_std
    mean_std[['WSSN', 'sum_PFAS_mean', 'pred_sum_PFAS_mean', 'pred_sum_PFAS_std', 'pred_sum_PFAS_5th', 'pred_sum_PFAS_95th']].to_csv('results/gw_wells_PFAS_predictions_mean_std.csv', index=False, float_format='%.4f')
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

    logger.info(f"Number of samples after removing zero values: {len(pfas_gw)}")
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    # Plot the bars with error bars
    ax.bar(pfas_gw.index, pfas_gw['pred_sum_PFAS_mean'], yerr=pfas_gw['pred_sum_PFAS_std'], 
        capsize=4, color='lightblue', label='Predicted Mean ± Std')



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



def classify_uncertainty(pfas_uncertainty, logger): 
    """ using k-means to classify the uncertainty into classes based on the elbow method, using the following features:
    1- pred_sum_PFAS_mean 
    2 - pred_sum_PFAS_std 
    3 - pred_sum_PFAS_5th 
    4 - pred_sum_PFAS_95th
    """

    

    # Extract features for clustering
    X = pfas_uncertainty[['pred_sum_PFAS_mean', 'pred_sum_PFAS_std', 'pred_sum_PFAS_5th', 'pred_sum_PFAS_95th']].values

    # Determine the optimal number of clusters using the elbow method
    wcss = []
    for i in range(1, 11):  # Test cluster sizes from 1 to 10
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    ## first plot bounds
    
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig("results/elbow_method.png", dpi=300)
    plt.show()

    # Choose the optimal number of clusters (based on visual inspection of the elbow plot)
    optimal_clusters = 4 # Set this based on where the "elbow" occurs

    # Perform K-Means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(X)
    pfas_uncertainty['uncertainty_class'] = kmeans.labels_


    from sklearn.mixture import GaussianMixture

    # Assuming pfas_uncertainty is your DataFrame and optimal_clusters is defined
    X = pfas_uncertainty[['pred_sum_PFAS_mean', 'pred_sum_PFAS_std', 'pred_sum_PFAS_5th', 'pred_sum_PFAS_95th']].values
    gmm = GaussianMixture(n_components=optimal_clusters, random_state=0).fit(X)

    # Calculate probability of each cluster
    probabilities = gmm.predict_proba(X)

    # Extract unique probabilities for each class
    unique_probabilities = {i: probabilities[:, i].mean() for i in range(optimal_clusters)}

    # Log the unique probabilities
    logger.info(f"Unique probabilities for each class: {unique_probabilities}")
    # Assign colors to clusters
    #cluster_colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
    # create cluster_colors based on the optimal_clusters, starting with color red, orange, yellow, green, blue, purple
    cluster_colors = {i: plt.cm.tab10(i) for i in range(optimal_clusters)}

    pfas_uncertainty['color'] = pfas_uncertainty['uncertainty_class'].map(cluster_colors)

    # Plot the clusters
    fig, ax = plt.subplots(figsize=(8, 8))
    bounds = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl").to_crs("EPSG:4326")
    bounds.plot(facecolor='none', edgecolor='black', linewidth=1, alpha=0.5, ax=ax)
    pfas_uncertainty.to_crs("EPSG:4326").plot(ax=ax, color=pfas_uncertainty['color'], alpha=0.5)

    # Create a legend with custom patches
    legend_patches = [Patch(color=color, label=f'Cluster {i}') for i, color in cluster_colors.items()]
    ax.legend(handles=legend_patches, title="Uncertainty Clusters")
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('PFAS Uncertainty Clusters')
    plt.tight_layout()
    plt.savefig("results/uncertainty_clusters.png", dpi=300)
    plt.close()






def plot_uncertainty_distribution(pfas_gw, logger):
    pfas_gw = pfas_gw.to_crs("EPSG:4326")
    prediction_uncertainty = pd.read_csv("results/gw_wells_PFAS_predictions_mean_std.csv")
    pfas_uncertainty = pfas_gw[['WSSN', "DEM_250m",'geometry']].merge(prediction_uncertainty, on='WSSN', how='inner')
    bounds = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl").to_crs("EPSG:4326")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    maximum_95th = pfas_uncertainty['pred_sum_PFAS_95th'].max() + 1
    fig, ax = plt.subplots(figsize=(8, 8))

    color_dict = {
        "10": 'blue',
        "25": 'green',
        "50": 'yellow',
        "75": 'orange',
        "100": 'red',
        "125": 'purple',
        f"{maximum_95th:.1f}": 'black',
    }

    pfas_uncertainty['95th_class'] = pd.cut(
        pfas_uncertainty['pred_sum_PFAS_95th'],
        bins=[0, 10, 25, 50, 75, 100, 125, maximum_95th],
        labels=[
            "10",
            "25",
            "50",
            "75",
            "100",
            "125",
            f"{maximum_95th:.1f}",
        ],
    )
  
    ### calculate probability of each class
    counts = pfas_uncertainty['95th_class'].value_counts()
    probabilities = counts / counts.sum()
    logger.info(f"Probabilities of each class: {probabilities}")
    


    #pfas_gw.plot(ax=ax, color='gray', alpha=0.5)
    bounds.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

    # Plot each category separately
    for label, color in color_dict.items():
        subset = pfas_uncertainty[pfas_uncertainty['95th_class'] == label]
        subset.plot(ax=ax, color=color, label=label, alpha=0.5)

    # Create a legend with probilities of the respectiv class
    legend_patches = [Patch(color=color, label=f'{label} (%{100*probabilities[label]:.2f})') for label, color in color_dict.items()]

    
    ax.legend(handles=legend_patches, title="95th percentile threshold")


    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.title('GNNs Predicted PFAS Concentration Uncertainty')
    plt.tight_layout()
    plt.savefig("results/pfas_uncertainty.png", dpi=300)
    plt.show()

    return pfas_uncertainty



def plot_loss_histograms():
    path = "/home/rafieiva/MyDataBase/codes/PFAS_GNN/results/GridSearchResults.csv"
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
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.title('Train Loss Distribution (No Outliers)')
    plt.xlabel('Train Loss')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(df_no_outliers['val_loss'], bins=30, color='lightgreen')
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.title('Validation Loss Distribution (No Outliers)')
    plt.xlabel('Validation Loss')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(df_no_outliers['test_loss'], bins=30, color='lightcoral')
    plt.grid(axis='both', linestyle='--', alpha=0.5)
    plt.title('Test Loss Distribution (No Outliers)')
    plt.xlabel('Test Loss')
    plt.ylabel('Frequency')


    
    plt.tight_layout()
    plt.savefig('results/loss_histograms_no_outliers.png', dpi=300)
    plt.close()

def process_uncertainty_analysis(logger):
    
    args = {
        "data_dir": "/data/MyDataBase/HuronRiverPFAS",
        "gw_features": ["DEM_250m"],

    }
    uncertainty_analysis(logger)
    plot_loss_histograms()
    pfas_gw = load_pfas_gw(args['data_dir'], args['gw_features'], logger)
    pfas_uncertainty = plot_uncertainty_distribution(pfas_gw, logger)
    classify_uncertainty(pfas_uncertainty, logger)




def shap_analysis(logger):
    """
    A wrapper function for the model to make predictions on a subset of input data.
    """
    ## load a random model from the models directory
    import torch
    import shap
    args = {
        "data_dir": "/data/MyDataBase/HuronRiverPFAS",
        "gw_features": ["DEM_250m"],
        'pfas_gw_columns': ['sum_PFAS'],
        'pfas_sites_columns': ['Industry'],
        'distance_threshold': 500,
        'pfas_sw_station_columns' : ['sum_PFAS'],

    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    data, pfas_gw,pfas_sw = load_dataset(args,device, logger)
    model = torch.load(os.path.join("models",os.listdir("models")[0])   )

    explainer = shap.Explainer(model, data)
    shap_values = explainer(data)
    shap.plots.waterfall(shap_values[0])
    plt.savefig("results/shap_values.png", dpi=300)

if __name__ == "__main__":
    
    logger = setup_logging()
    #shap_analysis(logger)
    process_uncertainty_analysis(logger)


