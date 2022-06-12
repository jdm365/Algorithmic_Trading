Variable number of ids per era. Need to update ResnetMain.py to have fixed number of id channels and create additional arbitrary batching dimension.
New dims -> (batch_size, n_ids/channel_dim, n_channels, channel_dim, n_features)


Rethink Generally
