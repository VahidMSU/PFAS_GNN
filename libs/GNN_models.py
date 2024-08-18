import torch
from torch_geometric.nn import HeteroConv, GATConv, Linear, JumpingKnowledge, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn as nn
import torch.nn.functional as F


def get_model_by_name(name, in_channels_dict, out_channels, aggregation, **kwargs):
    models = {
        'MainGNNModel': MainGNNModel,
        'simple_GNNModel': simple_GNNModel,
        'relu': simple_GNNModel_with_ReLU,
        'tanh': simple_GNNModel_with_tahn,
        'parametric_relu_attention': GNNModel_with_PReLU_and_Attention,
        'leaky_relu_attention': GNNModel_with_leaky_ReLU_and_Attention,
        'leaky_relu': GNNModel_with_leaky_ReLU,
        'parametric_relu': simple_GNNModel_with_parametric_ReLU,
    }

    if name in models:
        return models[name](in_channels_dict, out_channels, aggregation, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")




class MainGNNModel(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation):
        super(MainGNNModel, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
        }, aggr=aggregation)
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = self.prelu(x['gw_wells'])  # Use PReLU after the linear layer
        return x
    

    
class simple_GNNModel(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation):
        super(simple_GNNModel, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
        }, aggr=aggregation)
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        return x

class simple_GNNModel_with_ReLU(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation):
        super(simple_GNNModel_with_ReLU, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
        }, aggr=aggregation)
        self.linear = nn.Linear(out_channels, 1)
        

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = F.relu(x['gw_wells'])  # Add ReLU activation after the linear layer
        return x


class simple_GNNModel_with_tahn(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation):
        super(simple_GNNModel_with_tahn, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
        }, aggr=aggregation)
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = torch.tanh(x['gw_wells'])  # Use Tanh activation after the linear layer
        return x


class GNNModel_with_leaky_ReLU_and_Attention(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, num_heads=4, dropout_rate=0.5):
        super(GNNModel_with_leaky_ReLU_and_Attention, self).__init__()
        
        # Attention mechanism for 'pfas_sites' to 'gw_wells' and self-loops
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): GATConv(in_channels_dict['pfas_sites'], out_channels, heads=num_heads, concat=False, add_self_loops=False),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): GATConv(in_channels_dict['pfas_sites'], out_channels, heads=num_heads, concat=False, add_self_loops=True),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
        }, aggr=aggregation)
        
        self.dropout = nn.Dropout(dropout_rate)  # Apply dropout
        self.linear_gw = nn.Linear(out_channels, 1)
        self.linear_sites = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        # Apply the first HeteroConv layer with attention for 'pfas_sites'
        x = self.conv1(x_dict, edge_index_dict)

        # Apply ReLU to the output of the convolution
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Apply dropout
        x = {key: self.dropout(x[key]) for key in x}

        # Apply linear transformation to 'gw_wells' and 'pfas_sites'
        x['gw_wells'] = self.linear_gw(x['gw_wells'])
        x['pfas_sites'] = self.linear_sites(x['pfas_sites'])

        # Apply Leaky ReLU after the linear layers
        x['gw_wells'] = F.leaky_relu(x['gw_wells'], negative_slope=0.01)
        x['pfas_sites'] = F.leaky_relu(x['pfas_sites'], negative_slope=0.01)

        return x



class GNNModel_with_PReLU_and_Attention(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, num_heads=4, dropout_rate=0.5):
        super(GNNModel_with_PReLU_and_Attention, self).__init__()
        
        # Attention mechanism for 'pfas_sites' to 'gw_wells' and self-loops
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): GATConv(in_channels_dict['pfas_sites'], out_channels, heads=num_heads, concat=False, add_self_loops=False),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): GATConv(in_channels_dict['pfas_sites'], out_channels, heads=num_heads, concat=False, add_self_loops=True),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
        }, aggr=aggregation)
        
        self.dropout = nn.Dropout(dropout_rate)  # Apply dropout
        self.linear_gw = nn.Linear(out_channels, 1)
        self.linear_sites = nn.Linear(out_channels, 1)
        
        # Parametric ReLU layers
        self.prelu_gw = nn.PReLU()
        self.prelu_sites = nn.PReLU()

    def forward(self, x_dict, edge_index_dict):
        # Apply the first HeteroConv layer with attention for 'pfas_sites'
        x = self.conv1(x_dict, edge_index_dict)

        # Apply ReLU to the output of the convolution
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Apply dropout
        x = {key: self.dropout(x[key]) for key in x}

        # Apply linear transformation to 'gw_wells' and 'pfas_sites'
        x['gw_wells'] = self.linear_gw(x['gw_wells'])
        x['pfas_sites'] = self.linear_sites(x['pfas_sites'])

        # Apply PReLU after the linear layers
        x['gw_wells'] = self.prelu_gw(x['gw_wells'])
        x['pfas_sites'] = self.prelu_sites(x['pfas_sites'])

        return x


class GNNModel_with_leaky_ReLU(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation):
        super(GNNModel_with_leaky_ReLU, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
        }, aggr=aggregation)
        
        self.linear_gw = nn.Linear(out_channels, 1)
        self.linear_sites = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        # Apply the first HeteroConv layer
        x = self.conv1(x_dict, edge_index_dict)
        
        # Apply ReLU to the output of the convolution
        x = {key: F.relu(x[key]) for key in x.keys()}
        
        # Apply linear transformation to 'gw_wells' and 'pfas_sites'
        x['gw_wells'] = self.linear_gw(x['gw_wells'])
        x['pfas_sites'] = self.linear_sites(x['pfas_sites'])
        
        # Apply Leaky ReLU after the linear layers
        x['gw_wells'] = F.leaky_relu(x['gw_wells'], negative_slope=0.001)
        x['pfas_sites'] = F.leaky_relu(x['pfas_sites'], negative_slope=0.001)
        
        return x



class simple_GNNModel_with_parametric_ReLU(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation):
        super(simple_GNNModel_with_parametric_ReLU, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels)
            ### add swl_differece
        }, aggr=aggregation)
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = self.prelu(x['gw_wells'])  # Use PReLU after the linear layer
        return x
