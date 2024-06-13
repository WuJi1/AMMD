from socket import IP_DROP_MEMBERSHIP
import torch
import torch.nn as nn

class HeteroMMDDistance(nn.Module):
    # Heterogeneous MMD Distance
    def __init__(self, cfg, kernel,) -> None:
        super().__init__()
        self.kernel = kernel
        self.cfg = cfg
        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.n_groups = cfg.model.mmd.num_groups
        if self.kernel == 'gaussian':
            self.alphas = [cfg.model.mmd.alphas ** k for k in range(-3, 2)] 

    @ staticmethod
    def compute_mmd(kernel_ss, kernel_qq, kernel_sq, beta, gamma):
        # if beta and gamma are None
        # unbiased empirical estimate
        # A kernel Two-Sample Test
        nq = kernel_qq.size(1)
        ns = kernel_ss.size(1)
        if beta is None:
            num_feat_s = kernel_ss.size(-1)
            kernel_ss = kernel_ss.flatten(-2)
            mmd_s = kernel_ss.sum(dim=-1) - kernel_ss[:, :, :, :, ::num_feat_s+1].sum(dim=-1)
            mmd_s = 1.0 / (num_feat_s * (num_feat_s - 1)) * mmd_s
            mmd_s = mmd_s.unsqueeze(1).expand(-1, nq, -1, -1)
            mmd_sq = 1.0 / num_feat_s * kernel_sq.sum(dim=-2)
        else:
            beta = beta.unsqueeze(-1)
            kernel_ss = kernel_ss.unsqueeze(1).expand(-1, nq, -1, -1, -1, -1)
            mmd_s = beta.transpose(-1, -2) @ kernel_ss @ beta
            mmd_s = mmd_s.squeeze(dim=-1).squeeze(dim=-1)

            # unbiased-like style
            # num_feat_s = kernel_ss.size(-1)
            # kernel_ss_ = kernel_ss.flatten(-2)[:, :, :, :, ::num_feat_s+1]
            # diag_sum = (beta.squeeze(-1) ** 2 * kernel_ss_).sum(-1)
            # mmd_s = mmd_s - diag_sum
            #####

            mmd_sq = (beta.transpose(-1, -2) @ kernel_sq).squeeze(-2)
            
        if gamma is None:
            num_feat_q = kernel_qq.size(-1)
            kernel_qq = kernel_qq.flatten(-2)
            mmd_q = kernel_qq.sum(dim=-1) - kernel_qq[:, :, :, :, ::num_feat_q+1].sum(dim=-1)
            mmd_q = 1.0 / (num_feat_q * (num_feat_q - 1)) * mmd_q
            mmd_q = mmd_q.unsqueeze(-1).expand(-1, -1, ns, -1)
            mmd_sq = 1.0 / num_feat_q * mmd_sq.sum(dim=-1)
        else:
            gamma = gamma.unsqueeze(-1)
            kernel_qq = kernel_qq.unsqueeze(2).expand(-1, -1, ns, -1, -1, -1)
            mmd_q = gamma.transpose(-1, -2) @ kernel_qq @ gamma
            mmd_q = mmd_q.squeeze(-1).squeeze(-1)

            # unbiased-like style
            # num_feat_q = kernel_qq.size(-1)
            # kernel_qq_ = kernel_qq.flatten(-2)[:, :, :, :, ::num_feat_q+1]
            # diag_sum = (gamma.squeeze(-1) ** 2 * kernel_qq_).sum(dim=-1)
            # mmd_q = mmd_q - diag_sum
            #####

            mmd_sq = (mmd_sq.unsqueeze(-2) @ gamma).squeeze(-1).squeeze(-1)
        mmd_dis = mmd_s + mmd_q - 2.0 * mmd_sq # b, nq, ns, ng
        mmd_dis = mmd_dis.mean(dim=-1)
        return mmd_dis

    def forward(self, support_xf, query_xf, beta=None, gamma=None):
        # support_xf: b, num_support, num_groups, num_feat_s, c
        # query_xf: b, num_query, num_groups, num_feat_q, c
        # beta: b, num_query, num_support, num_groups, num_feat_s
        # gamma: b, num_query, num_support, num_groups, num_feat_q
        
        # kernel_ss: b, num_support, num_groups, num_feat_s, num_feat_s
        # kernel_qq: b, num_query, num_groups, num_feat_q, num_feat_q
        # kernel_sq: b, num_query, num_support, num_groups, num_feat_s, num_feat_q
        if self.kernel == 'linear':
            kernel_ss, kernel_qq, kernel_sq = linear_kernel(support_xf, query_xf)
        elif self.kernel == 'gaussian':
            kernel_ss, kernel_qq, kernel_sq = multi_gaussian_kernel(support_xf, query_xf, alphas=self.alphas)
        else:
            raise KeyError('kernel is not supported')
        
        mmd_dis = self.compute_mmd(kernel_ss, kernel_qq, kernel_sq, beta, gamma) # b, num_query, num_support
        return mmd_dis
        
        

    
def linear_kernel(support_xf, query_xf):
    # https://github.com/jindongwang/transferlearning/blob/master/code/deep/DaNN/mmd.py
    # https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/statistics_diff.py
    # support_xf: b, num_support, num_groups, num_feat, c
    # query_xf: b, num_query, num_groups, num_feat, c
    # Consider linear time MMD with a linear kernel:
    # K(f(x), f(y)) = f(x)^Tf(y)
    # h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)

    kernel_ss = support_xf @ support_xf.transpose(-1, -2)
    kernel_qq = query_xf @ query_xf.transpose(-1, -2)
    ns = support_xf.size(1)
    nq = query_xf.size(1)
    support_xf_ = support_xf.unsqueeze(1).expand(-1, nq, -1, -1, -1, -1).contiguous()
    query_xf_ = query_xf.unsqueeze(2).expand(-1, -1, ns, -1, -1, -1).contiguous()
    kernel_qs = support_xf_ @ query_xf_.transpose(-1, -2)
    return kernel_ss, kernel_qq, kernel_qs


def multi_gaussian_kernel(support_xf, query_xf, alphas):
    # support_xf: b, num_support, num_groups, num_feat, c
    # query_xf: b, num_query, num_groups, num_feat, c
    b, ns, ng, nsf, c = support_xf.size()
    _, nq, _, nqf, _ = query_xf.size()

    distances_ss = torch.cdist(support_xf.view(-1, nsf, c), support_xf.view(-1, nsf, c)).view(b, ns, ng, nsf, nsf)
    distances_qq = torch.cdist(query_xf.view(-1, nqf, c), query_xf.view(-1, nqf, c)).view(b, nq, ng, nqf, nqf)
    support_xf_ = support_xf.unsqueeze(1).expand(-1, nq, -1, -1, -1, -1).contiguous()
    query_xf_ = query_xf.unsqueeze(2).expand(-1, -1, ns, -1, -1, -1).contiguous()
    distances_sq = torch.cdist(support_xf_.view(-1, nsf, c), query_xf_.view(-1, nqf, c)).view(b, nq, ns, ng, nsf, nqf)
    kernels_ss, kernels_qq, kernels_qs = None, None, None
    for alpha in alphas:
        kernels_ss_a, kernels_qq_a, kernels_qs_a = map(lambda x: torch.exp(- alpha * x ** 2), [distances_ss, distances_qq, distances_sq])
        if kernels_ss is None:
            kernels_ss, kernels_qq, kernels_qs = kernels_ss_a, kernels_qq_a, kernels_qs_a
        else:
            kernels_ss = kernels_ss + kernels_ss_a
            kernels_qq = kernels_qq + kernels_qq_a
            kernels_qs = kernels_qs + kernels_qs_a
    return kernels_ss, kernels_qq, kernels_qs
