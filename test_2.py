
class GNN_prelu_edge_combined_dim_reduction_normalization(nn.Module):
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
        gw_edge_attr_dim_pg = edge_attr_dict[('pfas_sites', 'distance', 'gw_wells')].shape[1]
        
        self.edge_reduce_dim = nn.ModuleDict({
            str(('pfas_sites', 'distance', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('gw_wells', 'distance', 'pfas_sites')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('pfas_sites', 'distance', 'sw_stations')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('sw_stations', 'distance', 'pfas_sites')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('sw_stations', 'distance', 'gw_wells')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
            str(('gw_wells', 'distance', 'sw_stations')): nn.Linear(gw_edge_attr_dim_pg, reduced_dim),
        })
        
        # Normalization layers for edge attributes
        self.edge_norm = nn.ModuleDict({
            edge_type_str: BatchNorm1d(reduced_dim) if self.use_batchnorm else LayerNorm(reduced_dim)
            for edge_type_str in self.edge_reduce_dim.keys()
        })
        
        # First convolutional layer with edge attributes and embeddings
        self.conv1 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('pfas_sites', 'distance', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('sw_stations', 'distance', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('sw_stations', 'distance', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'distance', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(reduced_dim, out_channels, reduced_dim, edge_embedding_dim),
        }, aggr=aggregation)

        # Second convolutional layer with edge attributes and embeddings (same structure as the first)
        self.conv2 = HeteroConv({
            ('pfas_sites', 'distance', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'distance', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('pfas_sites', 'distance', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('sw_stations', 'distance', 'pfas_sites'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('sw_stations', 'distance', 'gw_wells'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
            ('gw_wells', 'distance', 'sw_stations'): SAGEConvWithEdgeAttrAndEmbedding(out_channels, out_channels, reduced_dim, edge_embedding_dim),
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
