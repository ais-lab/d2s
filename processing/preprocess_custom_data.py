#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Aug 2024
@author: thuan.aislab@gmail.com
"""

import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), ".."))
from utils.read_write_model import read_model, read_images_text, read_images_binary
import utils.utils as uuls 
import h5py
import argparse 
from tqdm import tqdm
import numpy as np



def preprocessing(hloc_out_dir:str, list_test:str, out_dir:str, dataset:str, scene:str):
    """
    This function will generate train and test data of 3D feature positions 
    based on hloc toolbox. The output will be stored in the out_dir folder.
    """
    
    print("--------  Generating Training and Testing data ---------  ")
    
    with open(list_test,'r') as f:
        listnames_test = f.read().rstrip().split('\n')
    
    sfm_path = osp.join(hloc_out_dir, "sfm_superpoint+superglue")
        
    out_dir = uuls.makedir_OutScene(out_dir, dataset, scene)
    if out_dir is None:
        print(f"[INFOR] The output directory has been created, if you want to re-run, \
              please delete the folder: dataset\{dataset}")
        print("-------------- DONE -------------- \n")
        return 0
    
    features = h5py.File(osp.join(hloc_out_dir, "feats-superpoint-n4096-r1024.h5"), 'r')
    cameras, images, points3D = read_model(sfm_path)
    name2id = {image.name: i for i, image in images.items()}
    
    i = 0
    j = 0 
    for id_, image in tqdm(images.items()):
        t_name = image.name
        camera = uuls.camera2txt(cameras[sfm_id])
        if t_name in listnames_test:
            # test data.
            mode = "test"
            s_name = "test_" + str(i) + ".h5"
            sfm_id = name2id[t_name]
            pose = uuls.text_pose(name2id[sfm_id].tvec, name2id[sfm_id].qvec)
            i += 1
            with open(osp.join(out_dir, mode, "readme.txt"), "a") as wt:
                wt.write("{0} {1} {2} {3}\n".format(*[t_name, s_name, pose, camera]))
        else:
            try:
                # train data.
                mode = "train"
                s_name = "train_" + str(j) + ".h5"
                s_name3d = "label_" + str(j) + ".h5"
                sfm_id = name2id[t_name]
                pose = uuls.text_pose(images[sfm_id].tvec, images[sfm_id].qvec)
                
                p3D_ids = images[sfm_id].point3D_ids
                xys = images[sfm_id].xys
                
                if not p3D_ids.size > 0:
                    continue
                
                p3Ds = np.stack([points3D[ii].xyz if ii != -1 else 
                                 np.array([0,0,0]) for ii in p3D_ids], 0)
                errors = np.stack([points3D[ii].error if ii != -1 else 
                                 np.array(0) for ii in p3D_ids], 0)

                assert len(p3D_ids) == len(xys) == len(p3Ds) == len(errors)
                
                data_3D = {}
                data_3D = {"p3D_ids":p3D_ids, "xys": xys, "p3Ds": p3Ds, "errors":errors}
                with h5py.File(osp.join(out_dir, mode, "h5", s_name3d), "w") as fd: 
                    grp = fd.create_group(s_name3d.replace(".h5", ""))
                    for k, v in data_3D.items():
                        grp.create_dataset(k, data=v)
                j += 1
                with open(osp.join(out_dir, mode, "readme.txt"), "a") as wt:
                    wt.write("{0} {1} {2} {3} {4}\n".format(*[t_name, s_name, s_name3d, pose, camera]))
            except:
                continue
        data= {}
        for k,v in features[t_name].items():   
            data[k] = v.__array__()

        with h5py.File(osp.join(out_dir, mode, "h5", s_name), "w") as fd: 
            grp = fd.create_group(s_name.replace(".h5", ""))
            for k, v in data.items():
                grp.create_dataset(k, data=v)

    print("-------------- DONE -------------- \n")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_test', 
                        type=str, 
                        required=True, 
                        help="path to list_test.txt")
    parser.add_argument('--hloc_out_dir',
                        type=str,
                        required=True,
                        help="Directory where you store result after running hloc")
    parser.add_argument('--out_dir', 
                        type=str, 
                        default="../dataset",
                        help="Directory to store dataset after preprocess")
    parser.add_argument('--dataset', 
                        type=str, 
                        default="YourDataset",
                        help="Dataset name")
    parser.add_argument('--scene',
                        type=str,
                        default="a_scene_in_dataset",
                        help="Scene name")

    args = parser.parse_args()

    # End initializing the parameters
    preprocessing(args.hloc_out_dir, args.list_test, args.out_dir, args.dataset, args.scene)


if __name__ == "__main__":
    main(sys.argv)