import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


class Meta_Linear(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Meta_Linear, self).__init__(in_features, out_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast,
                           self.bias.fast)  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Meta_Linear, self).forward(x)
        return out


class Meta_Embedding(nn.Embedding):  # used in MAML to forward input with fast weight
    def __init__(self, num_embedding, embedding_dim):
        super(Meta_Embedding, self).__init__(num_embedding, embedding_dim)
        self.weight.fast = None

    def forward(self, x):
        if self.weight.fast is not None:
            out = F.embedding(x.astype('int64'), self.weight.fast, self._padding_idx, self._sparse)
        else:
            out = F.embedding(x.astype('int64'), self.weight, self._padding_idx, self._sparse)
        return out


class Emb(nn.Layer):
    def __init__(self, col_names, max_idxs, embedding_size=4, use_cuda=True):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        use_cuda: bool, True for gpu or False for cpu
        """
        super(Emb, self).__init__()
        self.static_emb = StEmb(col_names['static'], max_idxs['static'], embedding_size, use_cuda)
        self.ad_emb = StEmb(col_names['ad'], max_idxs['ad'], embedding_size, use_cuda)
        self.dynamic_emb = DyEmb(col_names['dynamic'], max_idxs['dynamic'], embedding_size, use_cuda)
        self.col_names = col_names
        self.col_length_name = [x + '_length' for x in col_names['dynamic']]

    def forward(self, x):
        static_emb = self.static_emb(x[self.col_names['static']])

        dynamic_emb = self.dynamic_emb(x[self.col_names['dynamic']], x[self.col_length_name])

        concat_embeddings = paddle.concat([static_emb, dynamic_emb], 1)
        ad_emb = self.ad_emb(x[self.col_names['ad']])

        return concat_embeddings, ad_emb


class DyEmb(nn.Layer):
    def __init__(self, fnames, max_idxs, embedding_size=4, use_cuda=True):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        method: 'avg' or 'sum'
        use_cuda: bool, True for gpu or False for cpu
        """
        super(DyEmb, self).__init__()

        self.fnames = fnames
        self.max_idxs = max_idxs
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda

        self.embeddings = nn.LayerList(
            [Meta_Embedding(max_idxs + 1, self.embedding_size) for max_idxs in self.max_idxs.values()])

    def masked_fill(self, x, mask, value):
        y = paddle.full(x.shape, value, x.dtype)
        return paddle.where(mask, y, x)

    def forward(self, dynamic_ids, dynamic_lengths):
        """
        input: relative id
        dynamic_ids: Batch_size * Field_size * Max_feature_size
        dynamic_lengths: Batch_size * Field_size
        return: Batch_size * Field_size * Embedding_size
        """
        concat_embeddings = []
        for i, key in enumerate(self.fnames):
            # B*M

            dynamic_lengths_tensor = paddle.to_tensor(dynamic_lengths[key + '_length'].values.astype(float))

            dynamic_ids_tensor = paddle.to_tensor(np.array(dynamic_ids[key].values.tolist()))

            batch_size = paddle.shape(dynamic_ids_tensor).item(0)
            # embedding layer B*M*E
            dynamic_embeddings_tensor = self.embeddings[i](dynamic_ids_tensor)

            dynamic_lengths_tensor = dynamic_lengths_tensor.unsqueeze(1)
            mask = (paddle.arange(paddle.shape(dynamic_embeddings_tensor).item(1)).unsqueeze(0).astype(
                float) < dynamic_lengths_tensor.unsqueeze(1))
            mask = mask.squeeze(1).unsqueeze(2)

            dynamic_embedding = self.masked_fill(dynamic_embeddings_tensor, mask == 0, 0)
            # return dynamic_embedding

            dynamic_lengths_tensor[dynamic_lengths_tensor == 0] = 1

            dynamic_embedding = (dynamic_embedding.sum(axis=1) / dynamic_lengths_tensor.astype('float32')).unsqueeze(1)
            concat_embeddings.append(paddle.reshape(dynamic_embedding, [batch_size, 1, self.embedding_size]))
        # B*F*E
        concat_embeddings = paddle.concat(concat_embeddings, 1)

        return concat_embeddings


class StEmb(nn.Layer):
    def __init__(self, col_names, max_idxs, embedding_size=4, use_cuda=True):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        use_cuda: bool, True for gpu or False for cpu
        """
        super(StEmb, self).__init__()
        self.col_names = col_names
        self.max_idxs = max_idxs
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        # initial layer
        self.embeddings = nn.LayerList(
            [Meta_Embedding(max_idx + 1, self.embedding_size) for max_idx in self.max_idxs.values()])

    def forward(self, static_ids):
        """
        input: relative id
        static_ids: Batch_size * Field_size
        return: Batch_size * Field_size * Embedding_size
        """
        concat_embeddings = []
        batch_size = static_ids.shape[0]
        static_ids_tensor_list = []
        for i, key in enumerate(self.col_names):
            # B*1
            # print("static_ids",static_ids)
            static_ids_tensor = paddle.to_tensor(static_ids[key].values.astype(float))
            static_ids_tensor_list.append(static_ids_tensor)
            static_embeddings_tensor = self.embeddings[i](static_ids_tensor)

            concat_embeddings.append(paddle.reshape(static_embeddings_tensor, [batch_size, 1, self.embedding_size]))
        # B*F*E
        concat_embeddings = paddle.concat(concat_embeddings, 1)
        return concat_embeddings


class MultiLayerPerceptron(nn.Layer):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = []
        for embed_dim in embed_dims:
            layers.append(Meta_Linear(input_dim, embed_dim))
            layers.append(nn.ReLU())
            input_dim = embed_dim
        if output_layer:
            layers.append(Meta_Linear(input_dim, 1))
        self.mlp = nn.LayerList(layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        out1 = self.mlp[0](x)
        out2 = self.mlp[1](out1)
        return out2


class WideAndDeepModel(nn.Layer):
    """
    A pytorch implementation of wide and deep learning.
    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, col_names, max_ids, embed_dim, mlp_dims, dropout, use_cuda, num_expert, num_output):
        super().__init__()
        self.embedding = Emb(col_names, max_ids, embed_dim, use_cuda)
        self.embed_output_dim = (len(col_names['static']) + len(col_names['dynamic'])) * embed_dim
        self.ad_embed_dim = embed_dim * (1 + len(col_names['ad']))
        expert = []
        for i in range(num_expert):
            expert.append(MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, False))
        self.mlp = nn.LayerList(expert)
        output_layer = []
        for i in range(num_output):
            output_layer.append(Meta_Linear(mlp_dims[-1], 1))
        self.output_layer = nn.LayerList(output_layer)

        self.attention_layer = nn.Sequential(Meta_Linear(self.ad_embed_dim, mlp_dims[-1]),
                                             nn.ReLU(),
                                             Meta_Linear(mlp_dims[-1], num_expert),
                                             nn.Softmax(axis=1))
        self.output_attention_layer = nn.Sequential(Meta_Linear(self.ad_embed_dim, mlp_dims[-1]),
                                                    nn.ReLU(),
                                                    Meta_Linear(mlp_dims[-1], num_output),
                                                    nn.Softmax(axis=1))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        emb, ad_emb = self.embedding(x)
        # print("emb",emb)
        ad_emb = paddle.concat([paddle.mean(emb, axis=1, keepdim=True), ad_emb], 1)  # 32 7 64
        # print("ad_",ad_emb)
        fea = 0
        att = self.attention_layer(paddle.reshape(ad_emb, [-1, self.ad_embed_dim]))  # 32 8

        for i in range(len(self.mlp)):
            fea += (att[:, i].unsqueeze(1) * self.mlp[i](paddle.reshape(emb, [-1, self.embed_output_dim])))

        att2 = self.output_attention_layer(paddle.reshape(ad_emb, [-1, self.ad_embed_dim]))
        result = 0
        for i in range(len(self.output_layer)):
            result += (att2[:, i].unsqueeze(1) * F.sigmoid(self.output_layer[i](fea)))

        return result.squeeze(1)