import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class DenseCL(nn.Module):
    '''DenseCL.
    Part of the code is borrowed from:
        "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".
    '''

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 loss_lambda=0.5,
                 **kwargs):
        super(DenseCL, self).__init__()
        # build함수에 따라, backbone, neck의 initializae된 클래스가 반환되며, sequential로 묶인다.
        # The backbone and neck module are intialized then, the backabone and neck class module are binded by nn.Sequential.
        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.encoder_k = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        
        self.backbone = self.encoder_q[0]
        # key encoder는 momentum으로 update되기 때문에 requires_grad=False로 만들어 학습이 되지않게 한다.
        # The parameters of the key encoder are freezed because these parameters are updated by momentum of paramters of query network.
        for param in self.encoder_k.parameters():
            param.requires_grad = False
            
        # contrastive loss class.
        # models/heads/contrastive_head.py
        self.head = builder.build_head(head)
        
        # weight initialization
        self.init_weights(pretrained=pretrained)

        self.queue_len = queue_len
        self.momentum = momentum
        
        # gloabl과 pixel level contrastive loss의 비율.
        # This parameter is about leveraging global and dense level losses
        self.loss_lambda = loss_lambda

        # create the queue
        # register_buffer: 모델의 구성 ex)layer로써 존재하지만 update되지 않음.
        # register_buffer: It is the part of the model. However, these parameters are not updated
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0) # These are normalized for calculating normalized cosine similiarty.
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # create the second queue for dense output/ dense key에 대한 dictionary도 만들어야함
        # In addtion to MoCo, the dictionary for dense features are needed.
        self.register_buffer("queue2", torch.randn(feat_dim, queue_len))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q[0].init_weights(pretrained=pretrained)
        self.encoder_q[1].init_weights(init_linear='kaiming')
        
        # queue의 paramter를 그대로 key의 parameter에 복사한다.
        # The parameters of queue network are copied to the key network.
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        
        # 모든 프로세스에서 key를 모음.
        # Gather all the key tensors from distributed process
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        
        # 현재 포인터가 가르키는곳.
        # current pointer of the dictionary.
        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # 포인터부터 batchsize만큼 enqueue, dequeue과정 진행.
        # enqueue dequeue from pointer with offset 
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue2_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        
        # 모든 프로세스들에서 텐서를 모음.
        # Gather all the tensors from all the distributed process
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        # 총 배치수를 max index로 설정하여 random permutation하여 index를 섞음
        # The range is 0 to the number of all the batch size. The index range is permutated randomly.
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # 섞은 index를 다시 분산된 환경에 복사시킴.
        # Permutated index list are copied to the all the distributed process.
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        # index를 unshuffle하기위한 인덱스를 저장해둠.
        # Save the index for unshuffling
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        
        # 분산된 환경에 permutation된 key index를 해당 gpu에 배당된 개수만큼 분배.
        # Distributed the shuffled index to the each process.
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        
        # return shuffled keys and unshuffle index.
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
            # N is the batch size, 2 means augmented query and key.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        
        # compute query features
        """
        self.encoder[1]
            neck=dict(
            type='DenseCLNeck',
            in_channels=2048,
            hid_channels=2048,
            out_channels=128,
            num_grid=None)
        """
        
        """
        # DenseCL/openselfsup/models/necks.py 
        class DenseCLNeck(nn.Module):
            '''The non-linear neck in DenseCL.
                Single and dense in parallel: fc-relu-fc, conv-relu-conv
            '''
            def __init__(self,
                         in_channels,
                         hid_channels,
                         out_channels,
                         num_grid=None):
                super(DenseCLNeck, self).__init__()

                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.mlp = nn.Sequential(
                    nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
                    nn.Linear(hid_channels, out_channels))

                self.with_pool = num_grid != None
                if self.with_pool:
                    self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
                self.mlp2 = nn.Sequential(
                    nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(hid_channels, out_channels, 1))
                self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

            def init_weights(self, init_linear='normal'):
                _init_weights(self, init_linear)

            def forward(self, x):
                assert len(x) == 1
                x = x[0]

                avgpooled_x = self.avgpool(x)
                avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

                if self.with_pool:
                    x = self.pool(x) # sxs
                    
                x = self.mlp2(x) # sxs: bxdxsxs
                avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1
                
                x = x.view(x.size(0), x.size(1), -1) # bxdxs^2
                avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd
                return [avgpooled_x, x, avgpooled_x2]
        """
        # q_b: 2048x7x7
        # q: mlp(average_pool(x)), q_grid:conv(x), q2: average_pool(x)
        q_b = self.encoder_q[0](im_q) # backbone features
        q, q_grid, q2 = self.encoder_q[1](q_b)  # queries: NxC:gloabl; NxCxS^2:dense
        q_b = q_b[0]
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)

        q = nn.functional.normalize(q, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q_b = nn.functional.normalize(q_b, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # key encoder update 부분
            # key encoder update.
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # shuffling 후,
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            
            # key를 통과시켜 key feature를 뽑음.
            k_b = self.encoder_k[0](im_k)
            k, k_grid, k2 = self.encoder_k[1](k_b)  # keys: NxC; NxCxS^2
            k_b = k_b[0]
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)

            k = nn.functional.normalize(k, dim=1) # gloabl: mlp(avgpool(x))
            k2 = nn.functional.normalize(k2, dim=1) # gloabl: avgpool(x) : it is dense negative key.
            k_grid = nn.functional.normalize(k_grid, dim=1) # projected dense
            k_b = nn.functional.normalize(k_b, dim=1) # backbone dense

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle)
            k_grid = self._batch_unshuffle_ddp(k_grid, idx_unshuffle)
            k_b = self._batch_unshuffle_ddp(k_b, idx_unshuffle)

        # compute logits
        """ gloabl part """
        # Einstein sum is more intuitive
        # positive logits: Nx1 (c=128), inner product. get similarity.
        # gloabl: mlp(avgpool(x))
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # negative logits: NxK, inner product. all corresponding between 'n' set in the batch and 'k' set in the dictionary.
        # negative는 queue에서 복재해서 similarity를 구한다.
        # negatives are from queue copied and detached.
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        """ dense part """
        # feat point set sim
        backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b) # (b,hw,hw=49) backbone dense
        # WTA(Winner Take All) to the key following the key axis based onthe query axis.
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1] # NxS^2 # query를 기준으로 key에 대한 WTA(Winner Take All) index를 얻는다.
        
        # positive index를 조회하여, 해당 위치에 있는 value들을 dim=2 축으로 쌓는다.
        # The values which are located in positive indice are concatenated with the axis 2.
        indexed_k_grid = torch.gather(k_grid, 2, densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1)) # NxCxS^2
        
        # inner product for dense contrastive loss.
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1) # NxS^2
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1) # NS^2X1
        
        # negative logits: NxK, inner product. all corresponding between 'n' set in the batch and 'k' set in the dictionary.
        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        l_neg_dense = torch.einsum('nc,ck->nk', [q_grid,
                                            self.queue2.clone().detach()])
        
        # calculate contrastive loss.
        loss_single = self.head(l_pos, l_neg)['loss_contra']
        loss_dense = self.head(l_pos_dense, l_neg_dense)['loss_contra']
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
            losses['loss_contra'] = self.criterion(logits, labels)
            return losses
        """
        losses = dict()
        losses['loss_contra_single'] = loss_single * (1 - self.loss_lambda)
        losses['loss_contra_dense'] = loss_dense * self.loss_lambda
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue2(k2)

        return losses

    def forward_test(self, img, **kwargs):
        im_q = img.contiguous()
        # compute query features
        #_, q_grid, _ = self.encoder_q(im_q)
        q_grid = self.backbone(im_q)[0]
        q_grid = q_grid.view(q_grid.size(0), q_grid.size(1), -1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        return None, q_grid, None

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
    # The dummy tensor is made.
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    # 모든 parallel process에 존재하는 key의 텐서를 가져와서 복제
    # The key tensor are copied in the all the parallel processes.
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    
    # key tensor들을 concat 
    output = torch.cat(tensors_gather, dim=0)
    return output
