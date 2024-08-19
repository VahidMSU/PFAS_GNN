import torch
from torch_geometric.nn import HeteroConv, GATConv, Linear, JumpingKnowledge, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn as nn
import torch.nn.functional as F


def get_model_by_name(name, in_channels_dict, out_channels, aggregation,edge_attr_dict, **kwargs):
    models = {
        'MainGNNModel': MainGNNModel,
        'simple': GNN_simple,
        'relu': GNN_relu,
        'tanh': GNN_tanh,
        'lrelu': GNN_lrelu,
        'prelu': GNN_prelu,
        'prelu_edge': GNN_prelu_edge,
        'prelu_attention': GNN_prelu_attention,
        'lrelu_attention': GNN_lrelu_attention,
    }

    if name in models:
        return models[name](in_channels_dict, out_channels, aggregation, edge_attr_dict, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")


class MainGNNModel(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None):
        super(MainGNNModel, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'distance', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
        }, aggr=aggregation)
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = self.prelu(x['gw_wells'])  # Use PReLU after the linear layer
        return x
    

    
class GNN_simple(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None):
        super(GNN_simple, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'distance', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
        }, aggr=aggregation)
        
        # Separate linear layers for gw_wells and sw_stations
        self.gw_wells_linear = nn.Linear(out_channels, 1)
        self.sw_stations_linear = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        
        # Apply the separate linear layers for gw_wells and sw_stations
        x['gw_wells'] = self.gw_wells_linear(x['gw_wells'])
        x['sw_stations'] = self.sw_stations_linear(x['sw_stations'])
        
        return x
class GNN_relu(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None):
        super(GNN_relu, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'distance', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
        }, aggr=aggregation)
        
        # Separate linear layers for gw_wells and sw_stations
        self.gw_wells_linear = nn.Linear(out_channels, 1)
        self.sw_stations_linear = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        
        # Apply the separate linear layers and ReLU for gw_wells and sw_stations
        x['gw_wells'] = F.relu(self.gw_wells_linear(x['gw_wells']))
        x['sw_stations'] = F.relu(self.sw_stations_linear(x['sw_stations']))
        
        return x



class GNN_tanh(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None):
        super(GNN_tanh, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'distance', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
        }, aggr=aggregation)
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = torch.tanh(x['gw_wells'])  # Use Tanh activation after the linear layer
        return x


class GNN_lrelu_attention(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation,edge_attr_dict=None, num_heads=4, dropout_rate=0.5):
        super(GNN_lrelu_attention, self).__init__()
        
        # Attention mechanism for 'pfas_sites' to 'gw_wells' and self-loops
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): GATConv(in_channels_dict['pfas_sites'], out_channels, heads=num_heads, concat=False, add_self_loops=False),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): GATConv(in_channels_dict['pfas_sites'], out_channels, heads=num_heads, concat=False, add_self_loops=True),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'distance', 'sw_stations'): GATConv(in_channels_dict['pfas_sites'], out_channels, heads=num_heads, concat=False, add_self_loops=False),
            ('sw_stations', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
        }, aggr=aggregation)
        
        self.dropout = nn.Dropout(dropout_rate)  # Apply dropout
        self.linear_gw = nn.Linear(out_channels, 1)
        self.linear_sites = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
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



class GNN_prelu_attention(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None, num_heads=4, dropout_rate=0.5):
        super(GNN_prelu_attention, self).__init__()
        
        # Attention mechanism for 'pfas_sites' to 'gw_wells' and self-loops
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): GATConv(in_channels_dict['pfas_sites'], out_channels, heads=num_heads, concat=False, add_self_loops=False),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): GATConv(in_channels_dict['pfas_sites'], out_channels, heads=num_heads, concat=False, add_self_loops=True),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'distance', 'sw_stations'): GATConv(in_channels_dict['pfas_sites'], out_channels, heads=num_heads, concat=False, add_self_loops=False),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
        }, aggr=aggregation)
        
        self.dropout = nn.Dropout(dropout_rate)  # Apply dropout
        self.linear_gw = nn.Linear(out_channels, 1)
        self.linear_sites = nn.Linear(out_channels, 1)
        
        # Parametric ReLU layers
        self.prelu_gw = nn.PReLU()
        self.prelu_sites = nn.PReLU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
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


class GNN_lrelu(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None):
        super(GNN_lrelu, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'distance', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
        }, aggr=aggregation)
        
        self.linear_gw = nn.Linear(out_channels, 1)
        self.linear_sites = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
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



class SAGEConvWithEdgeAttr(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super(SAGEConvWithEdgeAttr, self).__init__()
        self.sage_conv = SAGEConv(in_channels, out_channels)
        self.edge_transform = nn.Linear(in_features=edge_attr_dim, out_features=out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Apply transformation to edge attributes
        edge_attr_transformed = self.edge_transform(edge_attr)

        # Perform the SAGE convolution
        out = self.sage_conv(x, edge_index)

        # Add the edge attribute influence to the destination nodes
        row, col = edge_index
        out = out + torch.zeros_like(out).scatter_(0, col.unsqueeze(-1).expand_as(edge_attr_transformed), edge_attr_transformed)

        return out

class GNN_prelu_edge(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict):
        super(GNN_prelu_edge, self).__init__()
        
        # Automatically determine the dimension of edge attributes
        edge_attr_dim_pg = edge_attr_dict[('pfas_sites', 'distance', 'gw_wells')].shape[1]
        edge_attr_dim_gp = edge_attr_dict[('gw_wells', 'distance', 'pfas_sites')].shape[1]
        
        # Use the custom SAGEConvWithEdgeAttr that incorporates edge attributes
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConvWithEdgeAttr(in_channels_dict['pfas_sites'], out_channels, edge_attr_dim_pg),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConvWithEdgeAttr(in_channels_dict['gw_wells'], out_channels, edge_attr_dim_gp),
            ('pfas_sites', 'distance', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),

        }, aggr=aggregation)
        
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Use edge attributes during message passing
        x = self.conv1(x_dict, edge_index_dict, edge_attr_dict={
            ('pfas_sites', 'distance', 'gw_wells'): edge_attr_dict[('pfas_sites', 'distance', 'gw_wells')],
            ('gw_wells', 'distance', 'pfas_sites'): edge_attr_dict[('gw_wells', 'distance', 'pfas_sites')]
        })

        # Apply activation functions
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = self.prelu(x['gw_wells'])

        return x
class GNN_prelu(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None):
        super(GNN_prelu, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'distance', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'distance', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
        }, aggr=aggregation)
        
        # Separate linear layers for gw_wells and sw_stations
        self.gw_wells_linear = nn.Linear(out_channels, 1)
        self.sw_stations_linear = nn.Linear(out_channels, 1)
        
        # Separate PReLU activations for gw_wells and sw_stations
        self.gw_wells_prelu = nn.PReLU()
        self.sw_stations_prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        
        # Apply the separate linear layers and PReLU for gw_wells and sw_stations
        x['gw_wells'] = self.gw_wells_prelu(self.gw_wells_linear(x['gw_wells']))
        x['sw_stations'] = self.sw_stations_prelu(self.sw_stations_linear(x['sw_stations']))
        
        return x
