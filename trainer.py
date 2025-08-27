import os
import cv2
import torch
import random
import logging
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from prediction import test_calculate_metric

from utils.losses import DiceLoss

from model.CLIP.clip import clip
from model.CoOp import PromptMaker

from model.MC_CLIP import MC_Clip_3D


    
from skimage.measure import label
def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()
    
def get_cut_mask(out, thres=0.5, nms=1):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.best_performance = 0.0
        self.args = args

        # init clip
        self.clip, _ = clip.load("ViT-B/32", device=args.device)

        if "Pancreas" in args.dataset:
            prompts = ["background", "pancreas"]
        elif "BraTS" in args.dataset:
            prompts = ["background", "brain tumor"]
        elif "Lung" in args.dataset:
            prompts = ["background", "lung"]
            
        self.prompt_maker = PromptMaker(args=args,
                                        prompts=prompts,
                                        clip_model=self.clip,
                                        n_ctx=args.n_ctx,
                                        ).to(args.device)
        for name, param in self.prompt_maker.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
                

        # init Seg model
        self.model = MC_Clip_3D(args=args, n_channels=args.in_channels, n_classes=args.num_classes).to(args.device)

        self.eps = 1e-10
        self.optimizer = torch.optim.SGD([
                            {'params': self.model.parameters(), 'lr': args.base_lr, 'momentum': 0.9, 'weight_decay': 0.0001},
                            {'params': self.prompt_maker.parameters(), 'lr': args.base_lr, 'momentum': 0.9, 'weight_decay': 0.0001} 
                        ])

        
        self.dice_loss = DiceLoss(args.num_classes)
        self.ce_loss = nn.CrossEntropyLoss()

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)
    
    
    def get_entropy_map(self, p):
        ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
        return ent_map


    def train(self, volume_batch, label_batch, iter_num):
        txt_embed = self.prompt_maker().unsqueeze(0).float()
        outputs, feats = self.model(volume_batch, txt_embed)
        outputs_1, outputs_2 = outputs[0], outputs[1]
        
        # outputs_1_soft = torch.softmax(outputs_1, dim=1)
        # outputs_2_soft = torch.softmax(outputs_2, dim=1)
        
        outputs_1_sig = torch.sigmoid(outputs_1)
        outputs_2_sig = torch.sigmoid(outputs_2)
        
        # cog loss
        feats_1, feats_2 = feats[0], feats[1]
        feats_1 = feats_1 / feats_1.norm(dim=-1, keepdim=True)
        
        outputs_1_pred = F.interpolate(outputs_1_sig, size=(feats_1.shape[-3], feats_1.shape[-2], feats_1.shape[-1]), mode='trilinear', align_corners=False)
        vision_feats_1 = []
        for cls in range(outputs_1_sig.shape[1]):
            cls_img_feats = outputs_1_pred[:, cls, ...].unsqueeze(1) * feats_1
            vision_feats_1.append(cls_img_feats)
        vision_feats_1 = torch.stack(vision_feats_1, dim=1).mean(dim=(-3, -2, -1))
        
        outputs_2_pred = F.interpolate(outputs_2_sig, size=(feats_2.shape[-3], feats_2.shape[-2], feats_2.shape[-1]), mode='trilinear', align_corners=False)
        vision_feats_2 = []
        for cls in range(outputs_2_sig.shape[1]):
            cls_img_feats = outputs_2_pred[:, cls, ...].unsqueeze(1) * feats_2
            vision_feats_2.append(cls_img_feats)
        vision_feats_2 = torch.stack(vision_feats_2, dim=1).mean(dim=(-3, -2, -1))
        cog_loss = torch.mean((vision_feats_1 - txt_embed.repeat(outputs_1.shape[0], 1, 1)) ** 2) + torch.mean((vision_feats_2 - txt_embed.repeat(outputs_1.shape[0], 1, 1)) ** 2)
        
        # sup loss
        sup_loss_1 = self.ce_loss(outputs_1[:self.args.labeled_bs], label_batch[:self.args.labeled_bs]) + self.dice_loss(outputs_1_sig[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        sup_loss_2 = self.ce_loss(outputs_2[:self.args.labeled_bs], label_batch[:self.args.labeled_bs]) + self.dice_loss(outputs_2_sig[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
        sup_loss = sup_loss_1 + sup_loss_2
        
        # cons loss
        outputs_1_unlab_pseudoMap = get_cut_mask(outputs_1[self.args.labeled_bs:].clone().detach()).long()
        outputs_2_unlab_pseudoMap = get_cut_mask(outputs_2[self.args.labeled_bs:].clone().detach()).long()
        
        cons_loss_1 = self.ce_loss(outputs_1[self.args.labeled_bs:], outputs_2_unlab_pseudoMap) + self.dice_loss(outputs_1_sig[self.args.labeled_bs:], outputs_2_unlab_pseudoMap)
        cons_loss_2 = self.ce_loss(outputs_2[self.args.labeled_bs:], outputs_1_unlab_pseudoMap) + self.dice_loss(outputs_2_sig[self.args.labeled_bs:], outputs_1_unlab_pseudoMap)
        cons_loss = cons_loss_1 + cons_loss_2
        
        consistency_weight = self.get_current_consistency_weight(iter_num // 150)
        
        loss = sup_loss + self.args.cog_weight * cog_loss + consistency_weight * cons_loss
        
        if iter_num > self.args.mixed_iterations:
            labeled_volume_batch = volume_batch[:self.args.labeled_bs]
            unlabeled_volume_batch = volume_batch[self.args.labeled_bs:]
            labeled_label_batch = label_batch[:self.args.labeled_bs]
            
            if sup_loss_1 < sup_loss_2:
                lab_mixed_image = labeled_volume_batch * labeled_label_batch.unsqueeze(1) + unlabeled_volume_batch * (1 - labeled_label_batch).unsqueeze(1)
                unlab_mixed_image = unlabeled_volume_batch * outputs_1_unlab_pseudoMap.unsqueeze(1) + labeled_volume_batch * (1 - outputs_1_unlab_pseudoMap).unsqueeze(1)
                
                gt_lab_mixed_image = labeled_label_batch * labeled_label_batch + outputs_1_unlab_pseudoMap * (1 - labeled_label_batch)
                gt_unlab_mixed_image = outputs_1_unlab_pseudoMap * outputs_1_unlab_pseudoMap + labeled_label_batch * (1 - outputs_1_unlab_pseudoMap)
                
                mixed_image = torch.cat([lab_mixed_image, unlab_mixed_image], dim=0)
                gt_mixed_image = torch.cat([gt_lab_mixed_image, gt_unlab_mixed_image], dim=0)
            else:
                lab_mixed_image = labeled_volume_batch * labeled_label_batch.unsqueeze(1) + unlabeled_volume_batch * (1 - labeled_label_batch).unsqueeze(1)
                unlab_mixed_image = unlabeled_volume_batch * outputs_2_unlab_pseudoMap.unsqueeze(1) + labeled_volume_batch * (1 - outputs_2_unlab_pseudoMap).unsqueeze(1)
                
                gt_lab_mixed_image = labeled_label_batch * labeled_label_batch + outputs_2_unlab_pseudoMap * (1 - labeled_label_batch)
                gt_unlab_mixed_image = outputs_2_unlab_pseudoMap * outputs_2_unlab_pseudoMap + labeled_label_batch * (1 - outputs_2_unlab_pseudoMap)
                
                mixed_image = torch.cat([lab_mixed_image, unlab_mixed_image], dim=0)
                gt_mixed_image = torch.cat([gt_lab_mixed_image, gt_unlab_mixed_image], dim=0)

            outputs_mixed, feats_mixed = self.model(mixed_image, txt_embed)
            outputs_mixed_1, outputs_mixed_2 = outputs_mixed[0], outputs_mixed[1]
            
            outputs_mixed_1_sig = torch.sigmoid(outputs_mixed_1)
            outputs_mixed_2_sig = torch.sigmoid(outputs_mixed_2)
            
            if sup_loss_1 < sup_loss_2:
                mixed_sup_loss = self.ce_loss(outputs_mixed_2, gt_mixed_image) + self.dice_loss(outputs_mixed_2_sig, gt_mixed_image)
            else:
                mixed_sup_loss = self.ce_loss(outputs_mixed_1, gt_mixed_image) + self.dice_loss(outputs_mixed_1_sig, gt_mixed_image)
            
            
            # mixed cog loss
            feats_mixed_1, feats_mixed_2 = feats_mixed[0], feats_mixed[1]
            
            feats_mixed_1 = feats_mixed_1 / feats_mixed_1.norm(dim=-1, keepdim=True)
            outputs_mixed_1_pred = F.interpolate(outputs_mixed_1_sig, size=(feats_mixed_1.shape[-3], feats_mixed_1.shape[-2], feats_mixed_1.shape[-1]), mode='trilinear', align_corners=False)
            # outputs_mixed_1_pred = (F.interpolate(outputs_mixed_1_sig, size=(feats_mixed_1.shape[-3], feats_mixed_1.shape[-2], feats_mixed_1.shape[-1]), mode='trilinear', align_corners=False) > 0.5).float()
            vision_feats_mixed_1 = []
            for cls in range(outputs_mixed_1_sig.shape[1]):
                cls_img_feats = outputs_mixed_1_pred[:, cls, ...].unsqueeze(1) * feats_mixed_1
                vision_feats_mixed_1.append(cls_img_feats)
            vision_feats_mixed_1 = torch.stack(vision_feats_mixed_1, dim=1).mean(dim=(-3, -2, -1))
            
            feats_mixed_2 = feats_mixed_2 / feats_mixed_2.norm(dim=-1, keepdim=True)
            outputs_mixed_2_pred = F.interpolate(outputs_mixed_2_sig, size=(feats_mixed_2.shape[-3], feats_mixed_2.shape[-2], feats_mixed_2.shape[-1]), mode='trilinear', align_corners=False)
            # outputs_mixed_2_pred = (F.interpolate(outputs_mixed_2_sig, size=(feats_mixed_2.shape[-3], feats_mixed_2.shape[-2], feats_mixed_2.shape[-1]), mode='trilinear', align_corners=False) > 0.5).float()
            vision_feats_mixed_2 = []
            for cls in range(outputs_mixed_2_sig.shape[1]):
                cls_img_feats = outputs_mixed_2_pred[:, cls, ...].unsqueeze(1) * feats_mixed_2
                vision_feats_mixed_2.append(cls_img_feats)
            vision_feats_mixed_2 = torch.stack(vision_feats_mixed_2, dim=1).mean(dim=(-3, -2, -1))

            mixed_cog_loss = torch.mean((vision_feats_mixed_1 - txt_embed.repeat(outputs_mixed_1.shape[0], 1, 1)) ** 2) + torch.mean((vision_feats_mixed_2 - txt_embed.repeat(outputs_mixed_2.shape[0], 1, 1)) ** 2)
            
            loss = loss + mixed_sup_loss + self.args.cog_weight * mixed_cog_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        lr_ = self.args.base_lr * (1.0 - iter_num / self.args.max_iterations)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_
            
        logging.info('iteration %d : '
                     '  loss : %f'
                     '  sup_loss : %f'
                     '  cog_loss : %f'
                     '  cons_loss : %f'
                     '  lr_ : %10f'
                     '  consistency_weight : %10f'
                     % (iter_num, loss, sup_loss, self.args.cog_weight * cog_loss, consistency_weight * cons_loss,  lr_, consistency_weight))

    
    def test(self, snapshot_path, iter_num):
        self.model.eval()
        self.prompt_maker.eval()
        
        dice, hd95 = test_calculate_metric(self.args, self.model, self.prompt_maker, val=True)
        if dice > self.best_performance:
            self.best_performance = dice
            save_best = os.path.join(snapshot_path, 'Model_iter_' + str(iter_num) + ".pth")
            # save_best = os.path.join(snapshot_path, 'Best_Model.pth')
            torch.save({"model_dict": self.model.state_dict(), "prompt_dict": self.prompt_maker.prompt_learner.state_dict()}, save_best)
            
        self.model.train()
        self.prompt_maker.train()
        logging.info('iteration %d : '
                     '  mean_dice : %f '
                     '  mean_hd95 : %f '
                     % (iter_num, dice, hd95))

        



