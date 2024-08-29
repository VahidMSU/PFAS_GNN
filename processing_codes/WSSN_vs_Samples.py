
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
WSSN_huron = "/data/MyDataBase/HuronRiverPFAS/WSSN_Huron.pkl"
Huron_pfas_samples = "/data/MyDataBase/HuronRiverPFAS/Huron_PFAS_GW_Features.pkl"
Huron_river_bounds = "/data/MyDataBase/HuronRiverPFAS/Huron_River_basin_bound.pkl"
### plot both 
WSSN = pd.read_pickle(WSSN_huron).to_crs(epsg=4326)
samples = pd.read_pickle(Huron_pfas_samples).to_crs(epsg=4326)
bounds = pd.read_pickle(Huron_river_bounds).to_crs(epsg=4326)
ax, fig = plt.subplots(figsize=(6, 6))
bounds.plot(ax=fig, color='none', edgecolor='black')
WSSN.plot(ax=fig, color='blue', markersize=4)
samples.plot(ax=fig, color='red', markersize=9)
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title("GW Wells with WSSN and PFAS Samples in Huron River Basin")
plt.grid(alpha=0.5, linestyle='--', linewidth=0.5)
plt.legend(["Huron River Basin", "GW Wells with WSSN", "PFAS Samples"])
plt.tight_layout()
plt.legend([f"#{WSSN.WSSN.nunique()} Wells with unique WSSN", f"#{samples.WSSN.nunique()} Sampled Wells with unique WSSN"])
plt.savefig("figs/WSSN_wells_vs_samples_Huron.png", dpi=300)
plt.close()
