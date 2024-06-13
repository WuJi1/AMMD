import modules.registry as registry
from .mmd import AttentiveMMDPrompt,MMD,MMD_ori

################# MMD #################
@registry.Query.register("MMD_linear_triplet")
def mmd_gaussian_ce(channels, cfg):
    return MMD_ori(channels,cfg, loss='triplet', kernel='linear')

@registry.Query.register("MMD_linear_ce")
def mmd_gaussian_ce(channels, cfg):
    return MMD_ori(channels,cfg, loss='ce', kernel='linear')

################# AttentiveMMD #################
@registry.Query.register("AttenMMD_linear_triplet")
def mmd_gaussian_ce(channels, cfg):
    return AttentiveMMDPrompt(channels,cfg, loss='triplet', kernel='linear')

@registry.Query.register("AttenMMD_linear_ce")
def mmd_gaussian_ce(channels, cfg):
    return AttentiveMMDPrompt(channels,cfg, loss='ce', kernel='linear')

@registry.Query.register("AttenMMD_linear_nll")
def mmd_gaussian_ce(channels, cfg):
    return AttentiveMMDPrompt(channels,cfg, loss='NLL', kernel='linear')

@registry.Query.register("AttenMMD_linear_sup")
def mmd_gaussian_ce(channels, cfg):
    return AttentiveMMDPrompt(channels,cfg, loss='supcon', kernel='linear')

@registry.Query.register("AttenMMD_gaussian_triplet")
def mmd_gaussian_ce(channels, cfg):
    return AttentiveMMDPrompt(channels,cfg, loss='triplet', kernel='gaussian')

@registry.Query.register("AttenMMD_gaussian_ce")
def mmd_gaussian_ce(channels, cfg):
    return AttentiveMMDPrompt(channels,cfg, loss='ce', kernel='gaussian')

################# AttentiveMMDPrompt #################
@registry.Query.register("AttenMMDPrompt_linear_triplet")
def mmd_gaussian_ce(channels, cfg):
    return AttentiveMMDPrompt(channels,cfg, loss='triplet', kernel='linear',)

@registry.Query.register("AttenMMDPrompt_linear_ce")
def mmd_gaussian_ce(channels, cfg):
    return AttentiveMMDPrompt(channels,cfg, loss='ce', kernel='linear',)

@registry.Query.register("AttenMMDPrompt_gaussian_triplet")
def mmd_gaussian_ce(channels, cfg):
    return AttentiveMMDPrompt(channels,cfg, loss='triplet', kernel='gaussian')
