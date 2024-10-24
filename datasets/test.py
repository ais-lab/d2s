from pathlib import Path
import argparse
from data_collection import DataCollection 
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util.config as utilcfg
import util.visualize as u_vis
from omegaconf import OmegaConf

def parse_config():
    arg_parser = argparse.ArgumentParser(description='pre-processing for PL2Map dataset')
    arg_parser.add_argument('-d', '--dataset_dir', type=Path, default='datasets/imgs_datasets/', help='')
    arg_parser.add_argument('--sfm_dir', type=Path, default='datasets/gt_3Dmodels/', help='sfm ground truth directory')
    arg_parser.add_argument('--dataset', type=str, default="7scenes", help='dataset name')
    arg_parser.add_argument('-s', '--scene', type=str, default="office", help='scene name(s)')
    arg_parser.add_argument('-cp','--checkpoint', type=int, default=0, choices=[0,1], help='use pre-trained model')
    arg_parser.add_argument('--visdom', type=int, default=1,  choices=[0,1], help='visualize loss using visdom')
    arg_parser.add_argument('-c','--cudaid', type=int, default=0, help='specify cuda device id')
    arg_parser.add_argument('--use_depth', type=int, default=0, choices=[0,1], help='use SfM corrected by depth or not')
    arg_parser.add_argument('-o','--outputs', type=Path, default='logs/',
                        help='Path to the output directory, default: %(default)s')
    arg_parser.add_argument('-expv', '--experiment_version', type=str, default="00_00_00", help='experiment version folder')
    args, _ = arg_parser.parse_known_args()
    args.outputs = os.path.join(args.outputs, args.scene + "_" + args.experiment_version)
    print("Dataset: {} | Scene: {}".format(args.dataset, args.scene))
    cfg = utilcfg.load_config(f'cfgs/{args.dataset}.yaml', default_path='cfgs/default.yaml')
    cfg = OmegaConf.create(cfg)
    utilcfg.mkdir(args.outputs)

    # Save the config file for evaluation purposes
    config_file_path = os.path.join(args.outputs, 'config.yaml')
    OmegaConf.save(cfg, config_file_path)

    return args, cfg

def verify_two_keypoints(p2do, p2dof):
    # check if the two keypoints are the same
    count = 0
    wrong = ""
    matches1 = []
    matches2 = []
    for i in range(p2do.shape[0]):
        for ii in range(p2dof.shape[0]):
            if p2do[i][0] == p2dof[ii][0] and p2do[i][1] == p2dof[ii][1]:
                count += 1
                print(f"i: {i} | p2do: {p2do[i]} --- ii: {ii} | p2dof: {p2dof[ii]}")
                matches1.append(i)
                matches2.append(ii)
    print("Number of keypoints that are the same: ", count)
    print(p2do.shape, p2dof.shape)
    ids1 = [i for i in range(p2do.shape[0])]
    ids2 = [i for i in range(p2dof.shape[0])]
    wrong1 = [i for i in ids1 if i not in matches1]
    wrong2 = [i for i in ids2 if i not in matches2]
    print("Wrong keypoints in p2do: ", wrong1)
    print("Wrong keypoints in p2dof: ", wrong2)

def main():
    args, cfg = parse_config()
    __import__('pdb').set_trace()
    dataset = DataCollection(args, cfg, mode="train")
    # for i in range(100):
    img_name = dataset.train_imgs[0]
    print(img_name)
    
    # print(dataset.imgname2imgclass[img_name].camera.camera_array)
    # print(dataset.imgname2imgclass[img_name].pose.get_pose_vector())
    
    # u_vis.visualize_2d_points_lines_from_collection(dataset, img_name, mode="online")
    # u_vis.visualize_2d_lines_from_collection(dataset, img_name, mode="online")
    # u_vis.visualize_2d_lines_from_collection(dataset, img_name, mode="offline")
    # u_vis.open3d_vis_3d_points_from_datacollection(dataset)
    # u_vis.open3d_vis_3d_lines_from_single_imgandcollection(dataset, img_name)
    # u_vis.open3d_vis_3d_lines_from_datacollection(dataset)
    u_vis.visualize_2d_points_from_collection(dataset, img_name, mode="online")
    u_vis.visualize_2d_points_from_collection(dataset, img_name, mode="offline")
    # verify_two_keypoints(p2do, p2dof)
    # print(test)
    import pdb; pdb.set_trace()
    # dataset.image_loader(img_name, cfg.train.augmentation.apply, debug=True)
    # img_name = "seq-06/frame-000499.color.png"
    # train_img_list = dataset.train_imgs
    # i = 0
    # for img_name in train_img_list:
    #     i+=1
    #     if i%5 == 0:
    #         continue
    #     print(img_name)
    #     # u_vis.visualize_2d_points_from_collection(dataset, img_name, mode="offline")
    #     # u_vis.visualize_2d_points_from_collection(dataset, img_name, mode="online")
    #     u_vis.visualize_2d_lines_from_collection(dataset, img_name, mode="offline")
    #     # u_vis.visualize_2d_lines_from_collection(dataset, img_name, mode="online")
    #     # visualize 3D train lines
    #     # u_vis.open3d_vis_3d_lines_from_datacollection(dataset)
    #     if i > 2000:
    #         break
if __name__ == "__main__":
    main()











