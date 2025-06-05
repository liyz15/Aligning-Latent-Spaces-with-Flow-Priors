from .vq_loss import VQLoss
from .vq_flow_loss import VQFlowLoss
from .flow_head_loss import FlowHeadLoss
from .post_vq_loss import PostVQLoss

VQ_losses = {
    'VQLoss': VQLoss,
    'VQFlowLoss': VQFlowLoss,
    'FlowHeadLoss': FlowHeadLoss,
    'PostVQLoss': PostVQLoss,
}
