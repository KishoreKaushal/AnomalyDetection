import torch.nn as nn


class EntityEmbedding(nn.Module):
    """
        Parameters
        ----------
        embd_sizes : list of int
            Specify the list of sizes of the embedding vector
            for each categorical feature.

        sz_hidden_layers : list of int
            Specify the size of the hidden linear layers

        output_layer_sc : int
            Size of the output layers

        emb_layer_drop : float
            Dropout applied to the output of the embedding

        hidden_layer_drops : list of float
            List of dropout applied to the hidden layers

        use_bn : bool
            True, for batch normalization

    """
    def __init__(self,  embd_sizes, sz_hidden_layers, output_layer_sz,
                 emb_layer_drop, hidden_layer_drops, use_bn=False, ):
        super(EntityEmbedding, self).__init__()

        self.embds = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=s) for c,s in embd_sizes
        ])

        for embd in self.embds:
            embd.weight.data.uniform_(-1,1)

        # size of the vector after concatenating all the embedding layer
        conc_embd_size = sum(e.embedding_dim for e in self.embs)

        # linear layers followed by embedding layers
        # embedding layers --> linear layer 1 --> ... -> linear layer n

        sz_hidden_layers = [conc_embd_size] + sz_hidden_layers
        self.lins = nn.ModuleList([
            nn.Linear(sz_hidden_layers[i], sz_hidden_layers[i + 1])
            for i in range(len(sz_hidden_layers) - 1)
        ])

        # batch normalization layers after each linear layers
        # emb layer -> linear layer 1 -> batch norm 1 -> ... -> batch norm n-1 -> linear layer n
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in sz_hidden_layers[1:]
        ])

        # initializing hidden layers
        for out in self.lins:
            nn.init.kaiming_normal(out.weight.data)

        # initializing output layer
        self.output_layer = nn.Linear(sz_hidden_layers[-1], output_layer_sz)
        nn.init.kaiming_normal(self.output_layer.weight.data)

        self.emb_drop = nn.Dropout(emb_layer_drop)

        self.hidden_drops = nn.ModuleList([nn.Dropout(drop) for drop in hidden_layer_drops])


    def forward(self):
        pass




