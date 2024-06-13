import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

def _l2norm(x, dim=1, keepdim=True):
    return x / (1e-16 + torch.norm(x, 2, dim, keepdim))

def l2distance(x, y):
    """
    Input:
        x [.., c, M_x]
        y [.., c, M_y]
    Return:
        ret [.., M_x, M_y]
    """
    
    assert x.shape[:-2] == y.shape[:-2]
    prefix_shape = x.shape[:-2]

    c, M_x = x.shape[-2:]
    M_y = y.shape[-1]
    
    x = x.view(-1, c, M_x)
    y = y.view(-1, c, M_y)

    x_t = x.transpose(1, 2)
    x_t2 = x_t.pow(2.0).sum(-1, keepdim=True)
    y2 = y.pow(2.0).sum(1, keepdim=True)

    ret = x_t2 + y2 - 2.0 * x_t@y
    ret = ret.view(prefix_shape + (M_x, M_y))
    return ret

def batched_index_select(input_, dim, index):
    for ii in range(1, len(input_.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input_.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input_, dim, index)

def multihot_embedding(x, k, dim=-1):
    _, indice_ = torch.topk(x, k, dim=dim)
    shape_ = list(x.shape)
    shape_[dim] = k
    e = torch.zeros_like(x).scatter_(dim, indice_, torch.ones((shape_), device=x.device))
    return e

def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))

def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module

class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.
    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})
    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn


def triplet_loss(mmd_dis, query_y, thres):
    # pos_dis = torch.gather(mmd_dis, dim=-1, index=query_y.unsqueeze(-1))
    mask = one_hot(mmd_dis, query_y)
    
    #switch 1
    pos_dis = torch.masked_select(mmd_dis, mask)
    neg_dis = torch.masked_select(mmd_dis, ~mask).view(mask.size(0), -1).min(dim=-1)[0]

    #switch 2
    # pos_dis = torch.masked_select(mmd_dis, mask).view(mask.size(0), -1)
    # neg_dis = torch.masked_select(mmd_dis, ~mask).view(mask.size(0), -1)

    losses = F.relu(pos_dis - neg_dis + thres)
    return losses.mean()


def one_hot(mmd_dis, query_y):
    mask = torch.zeros_like(mmd_dis, dtype=torch.bool)
    return mask.scatter(1, query_y.unsqueeze(-1), 1)

def centering(support, query):
    # support: # *, c
    # query: *, c
    support = support - support.mean(dim=-1, keepdim=True)
    query = query - query.mean(dim=-1, keepdim=True)
    return support, query

class Metaprompt(nn.Module):
    def __init__(self, n_meta_prompt, feat_dim):
        super().__init__()
        self.meta_prompt = nn.Parameter(torch.empty(n_meta_prompt, feat_dim))
        # nn.init.orthogonal_(self.meta_prompt)
        nn.init.kaiming_uniform_(self.meta_prompt, a=math.sqrt(5.0))
        # self.dump_meta_prompt_ = None
    def forward(self):
        return self.meta_prompt


# def SupConLoss(features, labels, mask=None, temperature=0.07, contrast_mode='all', base_temperature=0.07):
#
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))
#
#         if len(features.shape) < 3:
#             raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                              'at least 3 dimensions are required')
#         if len(features.shape) > 3:
#             features = features.view(features.shape[0], features.shape[1], -1)
#
#         batch_size = features.shape[0]
#         if labels is not None and mask is not None:
#             raise ValueError('Cannot define both `labels` and `mask`')
#         elif labels is None and mask is None:
#             mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         elif labels is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             mask = torch.eq(labels, labels.T).float().to(device)
#         else:
#             mask = mask.float().to(device)
#
#         contrast_count = features.shape[1]
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
#         if contrast_mode == 'one':
#             anchor_feature = features[:, 0]
#             anchor_count = 1
#         elif contrast_mode == 'all':
#             anchor_feature = contrast_feature
#             anchor_count = contrast_count
#         else:
#             raise ValueError('Unknown mode: {}'.format(contrast_mode))
#
#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(anchor_feature, contrast_feature.T),
#             temperature)
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#
#         # tile mask
#         mask = mask.repeat(anchor_count, contrast_count)
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask
#
#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
#
#         # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#
#         # loss
#         loss = - (temperature / base_temperature) * mean_log_prob_pos
#         loss = loss.view(anchor_count, batch_size).mean()
#
#         return loss
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            features = features.unsqueeze(-1)
            # raise ValueError('`features` needs to be [bsz, n_views, ...],'
            #                  'at least 3 dimensions are required')
            # labels = labels.unsqueeze(0)
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss