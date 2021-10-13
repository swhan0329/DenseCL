import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class MOCO(nn.Module):
    """MOCO.

    Implementation of "Momentum Contrast for Unsupervised Visual
    Representation Learning (https://arxiv.org/abs/1911.05722)".
    Part of the code is borrowed from:
    "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        queue_len (int): Number of negative keys maintained in the queue.
            Default: 65536.
        feat_dim (int): Dimension of compact feature vectors. Default: 128.
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 **kwargs):
        super(MOCO, self).__init__()
        
        """
        model = dict(
        type='MOCO',
        pretrained=None,
        queue_len=65536,
        feat_dim=128,
        momentum=0.999,
        backbone=dict(
            type='ResNet',
            depth=50,
            in_channels=3,
            out_indices=[4],  # 0: conv-1, x: stage-x
            norm_cfg=dict(type='BN')),
        neck=dict(
            type='LinearNeck',
            in_channels=2048,
            out_channels=128,
            with_avg_pool=True),
        head=dict(type='ContrastiveHead', temperature=0.07))
        """
        # build함수에 따라, backbone, neck의 initializae된 클래스가 반환되며, sequential로 묶인다.
        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.encoder_k = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        
        self.backbone = self.encoder_q[0]
        # key encoder는 momentum으로 update되기 때문에 requires_grad=False로 만들어 학습이 되지않게 한다.
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        # contrastive loss head.
        # models/heads/contrastive_head.py
        self.head = builder.build_head(head)
        
        # weight initialization
        self.init_weights(pretrained=pretrained)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        # register_buffer: 모델의 구성 ex)layer로써 존재하지만 update되지 않음.
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q[0].init_weights(pretrained=pretrained)
        self.encoder_q[1].init_weights(init_linear='kaiming')
        
        # queue의 paramter를 그대로 key의 parameter에 복사한다.
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        
        # 모든 프로세스들에서 텐서를 모음.
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        # 총 배치수를 max index로 설정하여 random permutation하여 index를 섞음
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # 섞은 index를 다시 분산된 환경에 복사시킴.
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        # index를 unshuffle하기위한 인덱스를 저장해둠.
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        
        # 분산된 환경에 permutation된 key index를 해당 gpu에 배당된 개수만큼 분배.
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.
            여기서 N은 배치, 2는 augmented query, key의 개수를 표시한다.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        
        # compute query features
        q = self.encoder_q(im_q)[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1 (c=128), inner product. get similarity.
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # negative logits: NxK, inner product. all corresponding between 'n' set in the batch and 'k' set in the dictionary.
        # negative는 queue에서 복재해서 similarity를 구한다.
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Contrastive loss 계산
        """
            def forward(self, pos, neg):
            '''Forward head.
            Args:
                pos (Tensor): Nx1 positive similarity.
                neg (Tensor): Nxk negative similarity.
            Returns:
                dict[str, Tensor]: A dictionary of loss components.
            '''
            N = pos.size(0)
            logits = torch.cat((pos, neg), dim=1)
            logits /= self.temperature
            labels = torch.zeros((N, ), dtype=torch.long).cuda()
            losses = dict()
            losses['loss'] = self.criterion(logits, labels)
            return losses
        """
        losses = self.head(l_pos, l_neg)
        
        # 이번 step에서 뽑혔던 key를 enqueue하고, 제일 오래되었던 key를 dequeue해서 제거한다.
        self._dequeue_and_enqueue(k)

        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # 각 분산된 환경에 있는 key tensor의 shape과 같은 one tensor를 모아두는 리스트 생성.
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    # 모든 parallel process에 존재하는 key의 텐서를 가져와서 복제 
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    
    # key tensor들을 concat 
    output = torch.cat(tensors_gather, dim=0)
    return output
