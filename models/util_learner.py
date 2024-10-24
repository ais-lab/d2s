import torch.nn as nn 
import torch.optim as optim
import torch 

class CriterionPoint(nn.Module):
    '''
    # original implementation from https://arxiv.org/abs/2307.15250
    Criterion for point only'''
    def __init__(self, rpj_cfg, total_iterations=2000000):
        super(CriterionPoint, self).__init__()
        self.rpj_cfg = rpj_cfg
        self.total_iterations = total_iterations
        self.reprojection_loss = ReproLoss(total_iterations, self.rpj_cfg.soft_clamp, 
                                            self.rpj_cfg.soft_clamp_min, self.rpj_cfg.type, 
                                            self.rpj_cfg.circle_schedule)
        self.zero = fakezero()
    def forward(self, pred, target, iteration=2000000):
        batch_size, _, _ = pred['points3D'].shape
        validPoints = target["validPoints"]
        # get losses for points 
        square_errors_points = torch.norm((pred['points3D'][:,:3,:] - target["points3D"]), dim = 1)
        loss_points = torch.sum(validPoints*square_errors_points)/batch_size
        uncer_loss_points = torch.sum(torch.norm(validPoints - 1/(1+100*torch.abs(pred['points3D'][:,3,:])), dim = 1))/batch_size
        
        # get projection losses for points
        points_proj_loss = 0
        
        if self.rpj_cfg.apply:
            # get projection losses for points
            for i in range(batch_size): # default batch_size = 1
                prp_error, prp= project_loss_points(pred['keypoints'][i,:,:], pred['points3D'][i,:3,:], 
                                    target['pose'][i,:], target['camera'][i,:], validPoints[i,:])
                points_proj_loss += self.reprojection_loss.compute_point(prp_error, prp, iteration, validPoints[i,:])
            points_proj_loss = points_proj_loss / batch_size
        
        if iteration/self.total_iterations >= self.rpj_cfg.start_apply:
            total_loss = loss_points + uncer_loss_points + points_proj_loss
        else:
            total_loss = loss_points + uncer_loss_points
        points_proj_loss = self.zero if (isinstance(points_proj_loss, int) or isinstance(points_proj_loss, float)) else points_proj_loss
        return total_loss, loss_points, uncer_loss_points, self.zero, self.zero, points_proj_loss, self.zero
    

class fakezero(object):
	def __init__(self):
		pass
	def item(self):
		return 0


def qvec2rotmat(qvec):
    return torch.tensor([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def project_loss_points(gt_pt2Ds, pt3Ds, c_pose, camera, valids):
    '''
    gt_pt2Ds: 2xN
    pt3Ds: 3xN
    c_pose: 1x7
    camera: 1x5
    valids: 1xN
    '''
    device = pt3Ds.device
    R = qvec2rotmat(c_pose[3:]).to(device=device)
    t = torch.unsqueeze(c_pose[:3], dim = 1).to(device=device)
    if camera[0] == 0.0: # SIMPLE_PINHOLE
        fx = fy = camera[3] # focal length
        ppx = camera[4]
        ppy = camera[5]
    elif camera[0] == 1.0: # PINHOLE
        fx = camera[3] # focal length
        fy = camera[4]
        ppx = camera[5]
        ppy = camera[6]
    elif camera[0] == 2.0: # SIMPLE_RADIAL
        fx = fy = camera[3]
        ppx = camera[4]
        ppy = camera[5]
    else:
        raise f"Camera type {camera[0]} is not implemented"
    prd_2Ds = R@pt3Ds + t
    # project
    px = fx*prd_2Ds[0,:]/prd_2Ds[2,:] + ppx
    py = fy*prd_2Ds[1,:]/prd_2Ds[2,:] + ppy
    errors_x = (gt_pt2Ds[:,0] - px)**2
    errors_y = (gt_pt2Ds[:,1] - py)**2
    # return torch.mean(valids * torch.sqrt(errors_x + errors_y))
    return torch.sqrt(errors_x + errors_y), prd_2Ds # l2 distance error, and projected 2D points


def weighted_tanh(repro_errs, weight):
    # return weight * torch.tanh(repro_errs / weight).sum()
    return torch.mean(weight * torch.tanh(repro_errs / weight))

import numpy as np
class ReproLoss:
    """
    Original from: https://github.com/nianticlabs/ace
    Compute per-pixel reprojection loss using different configurable approaches.

    - tanh:     tanh loss with a constant scale factor given by the `soft_clamp` parameter (when a pixel's reprojection
                error is equal to `soft_clamp`, its loss is equal to `soft_clamp * tanh(1)`).
    - dyntanh:  Used in the paper, similar to the tanh loss above, but the scaling factor decreases during the course of
                the training from `soft_clamp` to `soft_clamp_min`. The decrease is linear, unless `circle_schedule`
                is True (default), in which case it applies a circular scheduling. See paper for details.
    - l1:       Standard L1 loss, computed only on those pixels having an error lower than `soft_clamp`
    - l1+sqrt:  L1 loss for pixels with reprojection error smaller than `soft_clamp` and
                `sqrt(soft_clamp * reprojection_error)` for pixels with a higher error.
    - l1+logl1: Similar to the above, but using log L1 for pixels with high reprojection error.
    """

    def __init__(self,
                 total_iterations,
                 soft_clamp=50,
                 soft_clamp_min=1,
                 type='dyntanh',
                 circle_schedule=True):

        self.total_iterations = total_iterations
        self.soft_clamp = soft_clamp
        self.soft_clamp_min = soft_clamp_min
        self.type = type
        self.circle_schedule = circle_schedule

    def compute_point(self, reprojection_error_b1, pred_cam_coords_b31, iteration, valids):
        
        # Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b31[2, :] < 0.1 # 0.1 is the min depth
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > 1000 # repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[2, :] > 1000 # 1000 is the max depth
        valids = valids.bool()
        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = (valids | invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_b1 = ~invalid_mask_b1

        # Reprojection error for all valid scene coordinates.
        repro_errs_b1N = reprojection_error_b1[valid_mask_b1] # valid_reprojection_error_b1
        return self.final_compute(repro_errs_b1N, iteration)

    def final_compute(self, repro_errs_b1N, iteration):

        if repro_errs_b1N.nelement() == 0:
            return 0

        if self.type == "tanh":
            return weighted_tanh(repro_errs_b1N, self.soft_clamp)

        elif self.type == "dyntanh":
            # Compute the progress over the training process.
            schedule_weight = iteration / self.total_iterations

            if self.circle_schedule:
                # Optionally scale it using the circular schedule.
                schedule_weight = 1 - np.sqrt(1 - schedule_weight ** 2)

            # Compute the weight to use in the tanh loss.
            loss_weight = (1 - schedule_weight) * self.soft_clamp + self.soft_clamp_min

            # Compute actual loss.
            return weighted_tanh(repro_errs_b1N, loss_weight)

        elif self.type == "l1":
            # L1 loss on all pixels with small-enough error.
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            return repro_errs_b1N[~softclamp_mask_b1].sum()

        elif self.type == "l1+sqrt":
            # L1 loss on pixels with small errors and sqrt for the others.
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            loss_l1 = repro_errs_b1N[~softclamp_mask_b1].sum()
            loss_sqrt = torch.sqrt(self.soft_clamp * repro_errs_b1N[softclamp_mask_b1]).sum()

            return loss_l1 + loss_sqrt

        else:
            # l1+logl1: same as above, but use log(L1) for pixels with a larger error.
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            loss_l1 = repro_errs_b1N[~softclamp_mask_b1].sum()
            loss_logl1 = torch.log(1 + (self.soft_clamp * repro_errs_b1N[softclamp_mask_b1])).sum()

            return loss_l1 + loss_logl1

#### Optimizer ####
class Optimizer:
    """
    Wrapper around torch.optim + learning rate
    """
    def __init__(self, params, nepochs,steps_per_epoch, **kwargs):
        
        self.nepochs = nepochs
        self.steps_per_epoch = steps_per_epoch
        self.base_lr = kwargs.pop("base_lr")
        self.method = kwargs.pop("method")
        self.max_lr = kwargs.pop("max_lr")
        self.apply_OneCycleLR = kwargs.pop("apply_OneCycleLR")
        self.apply_self_shrink = kwargs.pop("apply_self_shrink")
        assert self.apply_OneCycleLR != self.apply_self_shrink, "Only one of the learning rate scheduler can be applied"
        self.lr_decay_step = int(nepochs/kwargs.pop("num_lr_decay_step"))
        self.lr_decay = kwargs.pop('lr_decay')
        self.nfactor = 0
        self.lr = self.base_lr
        if self.method == 'sgd':
            print("OPTIMIZER: ---  sgd")
            self.learner = optim.SGD(params, lr=self.base_lr,
                                     weight_decay=kwargs.pop("weight_decay"), **kwargs)
        elif self.method == 'adam':
            print("OPTIMIZER: ---  adam")
            self.learner = optim.Adam(params, lr=self.base_lr,
                                      weight_decay=kwargs.pop("weight_decay"), **kwargs)
        elif self.method == 'rmsprop':
            print("OPTIMIZER: ---  rmsprop")
            self.learner = optim.RMSprop(params, lr=self.base_lr,
                                         weight_decay=kwargs.pop("weight_decay"), **kwargs)
        elif self.method == 'adamW':
            print("OPTIMIZER: ---  adamW")
            self.learner = optim.AdamW(params, lr=self.base_lr,
                                      weight_decay=kwargs.pop("weight_decay"), **kwargs)
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.learner,
                                                max_lr=self.max_lr,
                                                epochs=self.nepochs,
                                                steps_per_epoch=self.steps_per_epoch,
                                                cycle_momentum=False)

    def adjust_lr(self, epoch):
        # self-adjust learning rate
        ''' Adjust learning rate based on epoch.
        Optional: call this function if keep training the model after loading checkpoint
        '''
        if (self.method not in ['sgd', 'adam', 'adamW']) or (self.lr_decay_step == 0.0):
            return self.base_lr
        nfactor = epoch // self.lr_decay_step
        if nfactor > self.nfactor:
            decay_factor = (1-self.lr_decay)**nfactor
            self.lr = self.base_lr * decay_factor
            for param_group in self.learner.param_groups:
                param_group['lr'] = self.lr
        return self.lr

    def lr_scheduler(self, epoch=None):
        if self.apply_OneCycleLR:
            if self.learner._step_count < self.scheduler.total_steps:
                self.lr = self.scheduler.get_last_lr()[0]
                self.scheduler.step() 
        elif self.apply_self_shrink:
            assert epoch is not None, "Error! Epoch must be provided for self-shrink learning rate scheduler"
            self.lr = self.adjust_lr(epoch)
        else:
            raise ValueError("Error! Learning rate scheduler not supported.")
        return self.lr  
    
        
    
    

