import numpy as np
from omegaconf import OmegaConf
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from util.help_evaluation import getLine3D_from_modeloutput, getPoint3D_from_modeloutput
import time 
import poselib 

class Pose_Estimator():
    def __init__(self, localize_cfg, eval_cfg, spath):
        self.localize_cfg = localize_cfg # config file for localization
        self.eval_cfg = eval_cfg # local config for evaluation
        self.spath = spath
        self.uncertainty_point = eval_cfg.uncer_threshold_point
        self.pnppoint = eval_cfg.pnp_point
        if not self.eval_cfg.exist_results:
            self.checkexist()
    def checkexist(self):
        ''' 
        Check if the files exist, if yes, remove them
        '''
        trainfiles_list = ['est_poses_train_pointline.txt', 'est_poses_train_point.txt',
                       'gt_poses_train.txt']
        testfiles_list = ['est_poses_test_pointline.txt', 'est_poses_test_point.txt',
                       'gt_poses_test.txt']
        if self.eval_cfg.eval_train:
            self.rmfiles(trainfiles_list)
        if self.eval_cfg.eval_test:
            self.rmfiles(testfiles_list)

    def rmfiles(self, rm_list):
        for file in rm_list:
            if os.path.exists(os.path.join(self.spath, file)):
                os.remove(os.path.join(self.spath, file))

    def run(self, output, data, target, mode='train'):
        camera_pose_estimation(self.localize_cfg, output, data, target, self.spath, mode=mode,
                     uncertainty_point=self.uncertainty_point, pnppoint=self.pnppoint)

def camera_pose_estimation(localize_cfg, output, data, target, spath, mode='train', 
                           uncertainty_point=0.5, pnppoint=False):
    '''
    Creating same inputs for limap library and estimate camera pose
    '''
    p3ds_, point_uncer = getPoint3D_from_modeloutput(output['points3D'], uncertainty_point)
    p3ds = [i for i in p3ds_]
    p2ds = output['keypoints'][0].detach().cpu().numpy() + 0.5 # COLMAP
    num_extracted_points = len(p2ds)
    p2ds = p2ds[point_uncer,:]
    p2ds = [i for i in p2ds]
    camera = target['camera'][0].detach().cpu().numpy()
    if camera[0] == 0.0:
        camera_model = "SIMPLE_PINHOLE"
    elif camera[0] == 1.0:
        camera_model = "PINHOLE"
    elif camera[0] == 2.0:
        camera_model = "SIMPLE_RADIAL"
    else:
        raise ValueError("Error! Camera model not implemented.")
    poselibcamera = {'model': camera_model, 'width': camera[2], 'height': camera[1], 'params': camera[3:]}
    image_name = data['imgname'][0]
    
    if pnppoint:
        start = time.time()
        pose_point, info = poselib.estimate_absolute_pose(p2ds, p3ds, poselibcamera, {'max_reproj_error': 12.0}, {})
        est_time = time.time() - start
        num_inliers = info['num_inliers']
        num_points = len(info['inliers'])
        with open(os.path.join(spath, f"est_poses_{mode}_point.txt"), 'a') as f:
            f.write(f"{pose_point.t[0]} {pose_point.t[1]} {pose_point.t[2]} {pose_point.q[0]} {pose_point.q[1]} {pose_point.q[2]} {pose_point.q[3]} {est_time} {image_name} {num_inliers} {num_points} {num_extracted_points}\n")
    target_pose = target['pose'][0].detach().cpu().numpy()
    with open(os.path.join(spath, f"gt_poses_{mode}.txt"), 'a') as f:
        f.write(f"{target_pose[0]} {target_pose[1]} {target_pose[2]} {target_pose[3]} {target_pose[4]} {target_pose[5]} {target_pose[6]}\n")