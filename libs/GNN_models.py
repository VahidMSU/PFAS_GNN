import torch
from torch_geometric.nn import HeteroConv, SAGEConv, BatchNorm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F



def get_model_by_name(name, in_channels_dict, out_channels, aggregation,edge_attr_dict, **kwargs):
    models = {
        'MainGNNModel': MainGNNModel,
        'simple': GNN_simple,
        'relu': GNN_relu,
        'tanh': GNN_tanh,
        'prelu': GNN_prelu,
        'prelu_edge': GNN_prelu_edge,
        'prelu_edge_combined': GNN_prelu_edge_combined,
        'prelu_edge_attention': GNN_prelu_edge_attention,
        
    }

    if name in models:
        return models[name](in_channels_dict, out_channels, aggregation, edge_attr_dict, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")



class SAGEConvWithEdgeAttrAndAttention(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super(SAGEConvWithEdgeAttrAndAttention, self).__init__()
        self.sage_conv = SAGEConv(in_channels, out_channels)

        # Attention mechanism for edge attributes
        # The input dimension should be (out_channels + out_channels)
        self.attention = nn.Linear(out_channels + out_channels, 1)

        # Transformation of edge attributes
        self.edge_transform = nn.Linear(edge_attr_dim, out_channels)
        
        # Normalization layer
        self.norm = nn.BatchNorm1d(out_channels)
        self.residual = nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        # Perform the SAGE convolution
        out = self.sage_conv(x, edge_index)

        # Transform edge attributes
        edge_attr_transformed = self.edge_transform(edge_attr)

        # Calculate attention scores
        row, col = edge_index
        combined_features = torch.cat([out[col], edge_attr_transformed], dim=-1)

        # Adjust the check to reflect the actual combined dimension
        assert combined_features.shape[-1] == self.attention.in_features, \
            f"Expected input dimension for attention: {self.attention.in_features}, but got {combined_features.shape[-1]}"

        attention_weights = torch.sigmoid(self.attention(combined_features))

        # Apply attention to edge attributes
        edge_contributions = attention_weights * edge_attr_transformed
        out = out + torch.zeros_like(out).scatter_(0, col.unsqueeze(-1).expand_as(edge_contributions), edge_contributions)

        # Apply normalization and residual connection
        out = self.norm(out)
        out = self.residual(out) + out

        return F.relu(out)

class GNN_prelu_edge_attention(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict, reduced_dim=32, use_batchnorm=True):
        super(GNN_prelu_edge_attention, self).__init__()
        
        # Choose normalization type
        self.use_batchnorm = use_batchnorm

        # Dimensionality reduction for node features (without normalization on the original features)
        self.node_reduce_dim = nn.ModuleDict({
            node_type: nn.Linear(in_channels, reduced_dim) for node_type, in_channels in in_channels_dict.items()
        })
        
        # Normalization layers for reduced node features (only after dimensionality reduction)
        self.node_norm = nn.ModuleDict({
            node_type: nn.BatchNorm1d(reduced_dim) if self.use_batchnorm else nn.LayerNorm(reduced_dim) 
            for node_type in in_channels_dict.keys()
        })
        
        # Dimensionality reduction and normalization for edge attributes
        gw_edge_attr_dim_pg = edge_attr_dict[('pfas_sites', 'dis_edge', 'gw_wells')].shape[1]
        
        self.edge_reduce_dim = nn.ModuleDict({
            str(('pfas_sites', 'dis_edge', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('pfas_sites', 'dis_edge', 'sw_stations')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),

            str(('sw_stations', 'dis_edge', 'pfas_sites')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('sw_stations', 'dis_edge', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),

            str(('gw_wells', 'dis_edge', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),

            str(('gw_wells', 'dis_edge', 'pfas_sites')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('gw_wells', 'dis_edge', 'sw_stations')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('gw_wells', 'dis_edge', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
        })
        
        # Normalization layers for edge attributes
        self.edge_norm = nn.ModuleDict({
            edge_type_str: nn.BatchNorm1d(reduced_dim) if self.use_batchnorm else nn.LayerNorm(reduced_dim)
            for edge_type_str in self.edge_reduce_dim.keys()
        })
        
        # First convolutional layer with edge attributes and attention
        self.conv1 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndAttention(reduced_dim, out_channels, reduced_dim),

            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(reduced_dim, out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(reduced_dim, out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(reduced_dim, out_channels), 

        }, aggr=aggregation)

        # Second convolutional layer with edge attributes and attention (same structure as the first)
        self.conv2 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),

            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),

            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndAttention(out_channels, out_channels, reduced_dim),
            
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(out_channels, out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(out_channels, out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(out_channels, out_channels),
        }, aggr=aggregation)
        
        # Linear layers and PReLU activation
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Apply dimensionality reduction for node features (without normalization)
        x_dict = {node_type: self.node_reduce_dim[node_type](x) for node_type, x in x_dict.items()}
        
        # Apply normalization after dimensionality reduction for node features
        x_dict = {node_type: F.relu(self.node_norm[node_type](x)) for node_type, x in x_dict.items()}
        
        # Dimensionality reduction and normalization for edge attributes
        edge_attr_dict = {
            edge_type: F.relu(self.edge_norm[str(edge_type)](self.edge_reduce_dim[str(edge_type)](edge_attr))) 
            for edge_type, edge_attr in edge_attr_dict.items()
        }
        
        # First convolutional layer
        x = self.conv1(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

        # Apply activation after first layer
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Second convolutional layer
        x = self.conv2(x, edge_index_dict, edge_attr_dict=edge_attr_dict)

        # Apply activation after second layer
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Apply linear layer and PReLU activation
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = self.prelu(x['gw_wells'])

        x['sw_stations'] = self.linear(x['sw_stations'])
        x['sw_stations'] = self.prelu(x['sw_stations'])

        return x


class SAGEConvWithEdgeAttrAndEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim, edge_embedding_dim=16):

        super(SAGEConvWithEdgeAttrAndEmbedding, self).__init__()
        self.sage_conv = SAGEConv(in_channels, out_channels)
        
        # Edge embedding layer (can be an embedding layer or a small neural network)
        self.edge_embedding = nn.Linear(edge_attr_dim, edge_embedding_dim)
        
        # Transform edge embeddings to match the output dimension
        self.edge_transform_emb = nn.Linear(edge_embedding_dim, out_channels)
        
        # Direct edge attribute transformation
        self.edge_transform_attr = nn.Linear(edge_attr_dim, out_channels)
        
        self.gate = nn.Linear(out_channels * 3, out_channels)  # Update to reflect combined feature size
        self.norm = nn.BatchNorm1d(out_channels)
        self.residual = nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        # Compute edge embeddings
        edge_embeddings = F.relu(self.edge_embedding(edge_attr))
        
        # Transform edge embeddings to match the output dimension
        edge_attr_transformed_emb = self.edge_transform_emb(edge_embeddings)

        # Directly transform edge attributes
        edge_attr_transformed_attr = self.edge_transform_attr(edge_attr)
        
        # Combine both transformed edge features
        edge_attr_transformed = edge_attr_transformed_emb + edge_attr_transformed_attr

        # Perform the SAGE convolution
        out = self.sage_conv(x, edge_index)

        # Add the edge attribute influence to the destination nodes
        row, col = edge_index
        combined_features = torch.cat([out[col], edge_attr_transformed, edge_attr_transformed_attr], dim=-1)  # Updated combined features
        
        # Compute gating values
        gate_values = torch.sigmoid(self.gate(combined_features))
        
        # Weight the node features by the gate values
        edge_contributions = gate_values * edge_attr_transformed
        out = out + torch.zeros_like(out).scatter_(0, col.unsqueeze(-1).expand_as(edge_contributions), edge_contributions)
        
        # Apply normalization and residual connection
        out = self.norm(out)
        out = self.residual(out) + out

        return F.relu(out)
    


class GNN_prelu_edge_combined(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict, edge_embedding_dim=16, reduced_dim=32, use_batchnorm=True):
        super(GNN_prelu_edge_combined, self).__init__()
        
        # Choose normalization type
        self.use_batchnorm = use_batchnorm

        # Dimensionality reduction for node features (without normalization on the original features)
        self.node_reduce_dim = nn.ModuleDict({
            node_type: nn.Linear(in_channels, reduced_dim) for node_type, in_channels in in_channels_dict.items()
        })
        
        # Normalization layers for reduced node features (only after dimensionality reduction)
        self.node_norm = nn.ModuleDict({
            node_type: BatchNorm1d(reduced_dim) if self.use_batchnorm else LayerNorm(reduced_dim) 
            for node_type in in_channels_dict.keys()
        })
        
        # Dimensionality reduction and normalization for edge attributes
        gw_edge_attr_dim_pg = edge_attr_dict[('pfas_sites', 'dis_edge', 'gw_wells')].shape[1]
        
        self.edge_reduce_dim = nn.ModuleDict({
            str(('pfas_sites', 'dis_edge', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('pfas_sites', 'dis_edge', 'sw_stations')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),

            str(('sw_stations', 'dis_edge', 'pfas_sites')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('sw_stations', 'dis_edge', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),

            str(('gw_wells', 'dis_edge', 'pfas_sites')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('gw_wells', 'dis_edge', 'sw_stations')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('gw_wells', 'dis_edge', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),

        })
        
        # Normalization layers for edge attributes
        self.edge_norm = nn.ModuleDict({
            edge_type_str: BatchNorm1d(reduced_dim) if self.use_batchnorm else LayerNorm(reduced_dim)
            for edge_type_str in self.edge_reduce_dim.keys()
        })
        
        # First convolutional layer with edge attributes and embeddings
        self.conv1 = HeteroConv({

            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),

            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(reduced_dim, out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(reduced_dim, out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(reduced_dim, out_channels), 

        }, aggr=aggregation)

        # Second convolutional layer with edge attributes and embeddings (same structure as the first)
        self.conv2 = HeteroConv({

            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),

            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('sw_stations', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),

            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(out_channels, out_channels),
            ('sw_stations', 'self_loop', 'sw_stations'): SAGEConv(out_channels, out_channels),
            ('pfas_sites', 'self_loop', 'pfas_sites'): SAGEConv(out_channels, out_channels),
        }, aggr=aggregation)
        
        # Linear layers and PReLU activation
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # Apply dimensionality reduction for node features (without normalization)
        x_dict = {node_type: self.node_reduce_dim[node_type](x) for node_type, x in x_dict.items()}
        
        # Apply normalization after dimensionality reduction for node features
        x_dict = {node_type: F.relu(self.node_norm[node_type](x)) for node_type, x in x_dict.items()}
        
        # Dimensionality reduction and normalization for edge attributes
        edge_attr_dict = {
            edge_type: F.relu(self.edge_norm[str(edge_type)](self.edge_reduce_dim[str(edge_type)](edge_attr))) 
            for edge_type, edge_attr in edge_attr_dict.items()
        }
        
        # First convolutional layer
        x = self.conv1(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

        # Apply activation after first layer
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Second convolutional layer
        x = self.conv2(x, edge_index_dict, edge_attr_dict=edge_attr_dict)

        # Apply activation after second layer
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Apply linear layer and PReLU activation
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = self.prelu(x['gw_wells'])

        x['sw_stations'] = self.linear(x['sw_stations'])
        x['sw_stations'] = self.prelu(x['sw_stations'])

        return x


class SAGEConvWithEdgeAttr(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super(SAGEConvWithEdgeAttr, self).__init__()
        self.sage_conv = SAGEConv(in_channels, out_channels)
        self.edge_transform = nn.Linear(edge_attr_dim, out_channels)
        self.gate = nn.Linear(out_channels * 2, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.residual = nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        # Apply transformation to edge attributes
        edge_attr_transformed = self.edge_transform(edge_attr)

        # Perform the SAGE convolution
        out = self.sage_conv(x, edge_index)

        # Add the edge attribute influence to the destination nodes
        row, col = edge_index
        combined_features = torch.cat([out[col], edge_attr_transformed], dim=-1)
        
        # Compute gating values
        gate_values = torch.sigmoid(self.gate(combined_features))
        
        # Weight the node features by the gate values
        edge_contributions = gate_values * edge_attr_transformed
        out = out + torch.zeros_like(out).scatter_(0, col.unsqueeze(-1).expand_as(edge_contributions), edge_contributions)
        
        # Apply normalization and residual connection
        out = self.norm(out)
        out = self.residual(out) + out

        return F.relu(out)


class GNN_prelu_edge(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict):
        super(GNN_prelu_edge, self).__init__()
        
        # Automatically determine the dimension of edge attributes
        gw_edge_attr_dim_pg = edge_attr_dict[('pfas_sites', 'dis_edge', 'gw_wells')].shape[1]
        gw_edge_attr_dim_gp = edge_attr_dict[('gw_wells', 'dis_edge', 'pfas_sites')].shape[1]
        sw_edge_attr_dim_pg = edge_attr_dict[('pfas_sites', 'dis_edge', 'sw_stations')].shape[1]
        sw_edge_attr_dim_gp = edge_attr_dict[('sw_stations', 'dis_edge', 'pfas_sites')].shape[1]
        gw_gw_edge_attr_dim = edge_attr_dict[('gw_wells', 'dis_edge', 'gw_wells')].shape[1]
        
        # First convolutional layer with edge attributes
        self.conv1 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttr(in_channels_dict['pfas_sites'], out_channels, gw_edge_attr_dim_pg),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttr(in_channels_dict['gw_wells'], out_channels, gw_edge_attr_dim_gp),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttr(in_channels_dict['pfas_sites'], out_channels, sw_edge_attr_dim_pg),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttr(in_channels_dict['sw_stations'], out_channels, sw_edge_attr_dim_gp),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttr(in_channels_dict['gw_wells'], out_channels, gw_gw_edge_attr_dim),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
        }, aggr="sum")

        # Second convolutional layer with edge attributes
        self.conv2 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttr(out_channels, out_channels, gw_edge_attr_dim_pg),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttr(out_channels, out_channels, gw_edge_attr_dim_gp),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConvWithEdgeAttr(out_channels, out_channels, sw_edge_attr_dim_pg),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConvWithEdgeAttr(out_channels, out_channels, sw_edge_attr_dim_gp),
            ('gw_wells', 'dis_edge', 'gw_wells'): SAGEConvWithEdgeAttr(out_channels, out_channels, gw_gw_edge_attr_dim),
            ('gw_wells', 'self_loop', 'gw_wells'): SAGEConv(out_channels, out_channels),
        }, aggr="sum")

        # Dynamic PReLU activations for each node type
        self.prelu_gw_wells = nn.PReLU(num_parameters=out_channels)
        self.prelu_sw_stations = nn.PReLU(num_parameters=out_channels)

        # Linear layers to reduce dimensionality to 1
        self.linear_gw_wells = nn.Linear(out_channels, 1)
        self.linear_sw_stations = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # First convolutional layer
        x = self.conv1(x_dict, edge_index_dict, edge_attr_dict={
            ('pfas_sites', 'dis_edge', 'gw_wells'): edge_attr_dict[('pfas_sites', 'dis_edge', 'gw_wells')],
            ('gw_wells', 'dis_edge', 'pfas_sites'): edge_attr_dict[('gw_wells', 'dis_edge', 'pfas_sites')],
            ('pfas_sites', 'dis_edge', 'sw_stations'): edge_attr_dict[('pfas_sites', 'dis_edge', 'sw_stations')],
            ('sw_stations', 'dis_edge', 'pfas_sites'): edge_attr_dict[('sw_stations', 'dis_edge', 'pfas_sites')],
            ('gw_wells', 'dis_edge', 'gw_wells'): edge_attr_dict[('gw_wells', 'dis_edge', 'gw_wells')],
        })

        # Apply activation after first layer
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Second convolutional layer
        x = self.conv2(x, edge_index_dict, edge_attr_dict={
            ('pfas_sites', 'dis_edge', 'gw_wells'): edge_attr_dict[('pfas_sites', 'dis_edge', 'gw_wells')],
            ('gw_wells', 'dis_edge', 'pfas_sites'): edge_attr_dict[('gw_wells', 'dis_edge', 'pfas_sites')],
            ('pfas_sites', 'dis_edge', 'sw_stations'): edge_attr_dict[('pfas_sites', 'dis_edge', 'sw_stations')],
            ('sw_stations', 'dis_edge', 'pfas_sites'): edge_attr_dict[('sw_stations', 'dis_edge', 'pfas_sites')],
            ('gw_wells', 'dis_edge', 'gw_wells'): edge_attr_dict[('gw_wells', 'dis_edge', 'gw_wells')],
        })

        # Apply activation after second layer
        x = {key: F.relu(x[key]) for key in x.keys()}

        # Apply dynamic PReLU activation
        x['gw_wells'] = self.prelu_gw_wells(x['gw_wells'])
        x['sw_stations'] = self.prelu_sw_stations(x['sw_stations'])

        # Apply linear layer to reduce dimensionality to 1
        x['gw_wells'] = self.linear_gw_wells(x['gw_wells'])
        x['sw_stations'] = self.linear_sw_stations(x['sw_stations'])

        return x





class MainGNNModel(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None):
        super(MainGNNModel, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
        }, aggr=aggregation)
        self.linear = nn.Linear(out_channels, 1)
        self.prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = self.prelu(x['gw_wells'])  # Use PReLU after the linear layer

        x['sw_stations'] = self.linear(x['sw_stations'])
        x['sw_stations'] = self.prelu(x['sw_stations'])
        
        return x
    

    
class GNN_simple(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None):
        super(GNN_simple, self).__init__()
        self.conv1 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
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
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
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
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
        }, aggr=aggregation)
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: F.relu(x[key]) for key in x.keys()}
        x['gw_wells'] = self.linear(x['gw_wells'])
        x['gw_wells'] = torch.tanh(x['gw_wells'])  # Use Tanh activation after the linear layer

        x['sw_stations'] = self.linear(x['sw_stations'])
        x['sw_stations'] = torch.tanh(x['sw_stations'])


        return x





class GNN_prelu(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggregation, edge_attr_dict=None, dropout_rate=0.5):
        super(GNN_prelu, self).__init__()
        
        # First HeteroConv layer
        self.conv1 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['gw_wells'], out_channels),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConv(in_channels_dict['pfas_sites'], out_channels),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConv(in_channels_dict['sw_stations'], out_channels),
        }, aggr=aggregation)

        # Second HeteroConv layer with same configuration but different parameters
        self.conv2 = HeteroConv({
            ('pfas_sites', 'dis_edge', 'gw_wells'): SAGEConv(out_channels, out_channels),
            ('gw_wells', 'dis_edge', 'pfas_sites'): SAGEConv(out_channels, out_channels),
            ('pfas_sites', 'dis_edge', 'sw_stations'): SAGEConv(out_channels, out_channels),
            ('sw_stations', 'dis_edge', 'pfas_sites'): SAGEConv(out_channels, out_channels),
        }, aggr=aggregation)

        # Batch normalization layers
        self.batch_norm_gw_wells = BatchNorm(out_channels)
        self.batch_norm_sw_stations = BatchNorm(out_channels)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Linear layers for final output
        self.gw_wells_linear = nn.Linear(out_channels, 1)
        self.sw_stations_linear = nn.Linear(out_channels, 1)
        
        # PReLU activations
        self.gw_wells_prelu = nn.PReLU()
        self.sw_stations_prelu = nn.PReLU()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # First convolutional layer
        x = self.conv1(x_dict, edge_index_dict)
        
        # Apply ReLU and batch normalization after the first convolution
        x['gw_wells'] = F.relu(self.batch_norm_gw_wells(x['gw_wells']))
        x['sw_stations'] = F.relu(self.batch_norm_sw_stations(x['sw_stations']))

        # Second convolutional layer
        x = self.conv2(x, edge_index_dict)
        
        # Apply ReLU and batch normalization after the second convolution
        x['gw_wells'] = F.relu(self.batch_norm_gw_wells(x['gw_wells']))
        x['sw_stations'] = F.relu(self.batch_norm_sw_stations(x['sw_stations']))

        # Apply dropout
        x['gw_wells'] = self.dropout(x['gw_wells'])
        x['sw_stations'] = self.dropout(x['sw_stations'])

        # Final linear layers and PReLU activations
        x['gw_wells'] = self.gw_wells_prelu(self.gw_wells_linear(x['gw_wells']))
        x['sw_stations'] = self.sw_stations_prelu(self.sw_stations_linear(x['sw_stations']))

        return x

