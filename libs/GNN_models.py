import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import HeteroConv


class MainGNNModel(nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super(MainGNNModel, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
            ### add swl_differece
        }, aggr='sum')
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = self.prelu(x['gw_wells'])  # Use PReLU after the linear layer
        return x


class simple_GNNModel(nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super(simple_GNNModel, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
        }, aggr='mean')
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        return x
    


class simple_GNNModel_with_ReLU(nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super(simple_GNNModel_with_ReLU, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
        }, aggr='mean')
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = F.relu(x['gw_wells'])  # Add ReLU activation after the linear layer
        return x




class simple_GNNModel_with_tahn(nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super(simple_GNNModel_with_tahn, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
        }, aggr='mean')
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = torch.tanh(x['gw_wells'])  # Use Tanh activation after the linear layer
        return x




class simple_GNNModel_with_leaky_ReLU(nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super(simple_GNNModel_with_leaky_ReLU, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
        }, aggr='mean')
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = F.leaky_relu(x['gw_wells'], negative_slope=0.01)  # Use Leaky ReLU after the linear layer
        return x




class simple_GNNModel_with_parametric_ReLU(nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super(simple_GNNModel_with_parametric_ReLU, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
            ### add swl_differece
        }, aggr='mean')
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = self.prelu(x['gw_wells'])  # Use PReLU after the linear layer
        return x
