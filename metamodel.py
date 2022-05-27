import paddle
from collections import OrderedDict
from paddle.nn import functional as F
import numpy as np
import copy
from model import WideAndDeepModel
from model import Meta_Linear
from model import Meta_Embedding


class MetaModel(paddle.nn.Layer):
    def __init__(self, col_names, max_ids, embed_dim, mlp_dims, dropout, use_cuda, local_lr, global_lr,
                 weight_decay, base_model_name, num_expert, num_output):
        super(MetaModel, self).__init__()
        self.model = WideAndDeepModel(col_names=col_names, max_ids=max_ids, embed_dim=embed_dim,
                                      mlp_dims=mlp_dims, dropout=dropout, use_cuda=use_cuda, num_expert=num_expert,
                                      num_output=num_output)
        self.local_lr = local_lr
        self.criterion = paddle.nn.BCELoss()
        self.meta_optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(), learning_rate=global_lr,
                                                    weight_decay=float(weight_decay))

    def forward(self, x):
        return self.model(x)

    def local_update(self, support_set_x, support_set_y):

        fast_parameters = list(self.model.parameters())
        for weight in fast_parameters:
            weight.fast = None

        support_set_y_pred = self.model(support_set_x)
        label = paddle.to_tensor(support_set_y.astype('float32'))
        loss = self.criterion(support_set_y_pred, label)

        self.model.clear_gradients()
        loss.backward()

        fast_parameters = list(self.model.parameters())
        for weight in fast_parameters:
            if weight.grad is None:
                continue
            if weight.fast is None:
                weight.fast = weight - self.local_lr * weight.grad  # create weight.fast
            else:
                weight.fast = weight.fast - self.local_lr * weight.grad

        self.model.clear_gradients()
        return loss

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys):
        batch_sz = len(support_set_xs)

        losses_q = []
        self.meta_optimizer.clear_grad()
        self.model.clear_gradients()
        for i in range(batch_sz):
            loss_sup = self.local_update(support_set_xs[i], support_set_ys[i])

            query_set_y_pred = self.model(query_set_xs[i])
            label = paddle.to_tensor(query_set_ys[i].astype('float32'))

            loss_q = self.criterion(query_set_y_pred, label)
            losses_q.append(loss_q)
        loss_average = paddle.stack(losses_q).mean(0)

        self.meta_optimizer.clear_grad()
        loss_average.backward()

        self.meta_optimizer.step()

        fast_parameters = list(self.model.parameters())
        for weight in fast_parameters:
            weight.fast = None

        return loss_average
