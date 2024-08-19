import hydrofunctions as hf

### get well information from the USGS NWIS for a specific WSSN id
def get_well_info(wssn):
    """
    Get well information from the USGS NWIS for a specific WSSN id
    """
    wssn = str(wssn)
    site = hf.NWIS(wssn)
    site.get_data()
    return site.df()
import pandas as pd

huron_river_wells = pd.read_pickle("/data/MyDataBase/HuronRiverPFAS/WSSN_Huron.pkl")
print(huron_river_wells.WSSN.values)    

wssn =  huron_river_wells.WSSN.values[0].astype(int).astype(str)
print(get_well_info(wssn))