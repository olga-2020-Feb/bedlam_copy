import os
import cv2
import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from smplx import SMPL
from ..core import constants, config
from ..core.config import DATASET_FILES, DATASET_FOLDERS
from ..utils.image_utils import crop, transform, random_crop, read_img


class DatasetHMR(Dataset):

    def __init__(self, options, dataset,
                 use_augmentation=True, is_train=True, num_images=0):
        super(DatasetHMR, self).__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = DATASET_FOLDERS[dataset]

        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(DATASET_FILES[is_train][dataset].replace('.npz', '-hands.npz').replace('all_npz_12_training','all_npz_12_hands'), allow_pickle=True)
        self.right_hand_detect = np.asarray(self.data['right_hand']).astype(np.bool)
        self.left_hand_detect = np.asarray(self.data['left_hand']).astype(np.bool)
        # Images with both left and right hand visible
        self.hand_detect = self.right_hand_detect * self.left_hand_detect

        self.imgname = self.data['imgname'][self.hand_detect]

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale'][self.hand_detect]
        self.center = self.data['center'][self.hand_detect]
        self.scale_l = self.data['scale_hand'][self.hand_detect, 0]
        self.center_l = self.data['center_hand'][self.hand_detect, 0]
        self.scale_r = self.data['scale_hand'][self.hand_detect, 1]
        self.center_r = self.data['center_hand'][self.hand_detect, 1]

        self.pose_cam = self.data['pose_cam'][self.hand_detect, :66].astype(np.float)
        self.lhand_pose = self.data['pose_cam'][self.hand_detect, 75:120].astype(np.float)
        self.rhand_pose = self.data['pose_cam'][self.hand_detect, 120:165].astype(np.float)
        num_joints = 127
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        self.betas = self.data['shape'][self.hand_detect].astype(np.float)

        if 'cam_int' in self.data:
            self.cam_int = self.data['cam_int'][self.hand_detect]

        if 'cam_ext' in self.data:
            self.cam_ext = self.data['cam_ext'][self.hand_detect]

        if 'trans_cam' in self.data:
            self.trans_cam = self.data['trans_cam'][self.hand_detect]
        # Get 2D keypoints

        if self.is_train:
            full_joints = self.data['gtkps'][self.hand_detect]
            self.keypoints = full_joints[:, :num_joints]
        else:
            try:
                full_joints = self.data['gtkps'][self.hand_detect]
                self.keypoints = full_joints[:, :num_joints]
            except KeyError:
                self.keypoints = np.zeros((len(self.imgname), num_joints, 3))

        if 'proj_verts' in self.data:
            self.proj_verts = self.data['proj_verts'][self.hand_detect]
        else:
            self.proj_verts = np.zeros((len(self.imgname), 437, 3))
        try:
            gender = self.data['gender'][self.hand_detect]
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        # evaluation variables
        if not self.is_train:
            self.joint_mapper_h36m = constants.H36M_TO_J17 if dataset == 'mpi-inf-3dhp' \
                else constants.H36M_TO_J14
            self.joint_mapper_gt = constants.J24_TO_J17 if dataset == 'mpi-inf-3dhp' \
                else constants.J24_TO_J14
            self.J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

            self.smpl_male = SMPL(config.SMPL_MODEL_DIR,
                                  gender='male',
                                  create_transl=False)
            self.smpl_female = SMPL(config.SMPL_MODEL_DIR,
                                    gender='female',
                                    create_transl=False)
        self.length = self.scale.shape[0]
        logger.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

    def rgb_processing(self, rgb_img_full, center, scale, img_res, kp2d=None):

        rgb_img = crop(rgb_img_full, center, scale, [img_res, img_res])
        rgb_img = np.transpose(rgb_img.astype('float32'),
                               (2, 0, 1)) / 255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale):
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                                   [self.options.IMG_RES,
                                   self.options.IMG_RES])
        kp[:, :-1] = 2. * kp[:, :-1] / self.options.IMG_RES - 1.
        kp = kp.astype('float32')
        return kp

    def scale_aug(self):
        sc = 1            # scaling
        if self.is_train:
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1 + self.options.SCALE_FACTOR,
                     max(1 - self.options.SCALE_FACTOR,
                     np.random.randn() * self.options.SCALE_FACTOR + 1))
        return sc

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()
        keypoints_orig = self.keypoints[index].copy()
        if self.options.proj_verts:
            proj_verts_orig = self.proj_verts[index].copy()
            item['proj_verts_orig'] = torch.from_numpy(proj_verts_orig).float()
            proj_verts = self.proj_verts[index].copy()
       
        # Apply scale augmentation
        sc = self.scale_aug()
        # apply crop augmentation
        if self.is_train and self.options.CROP_FACTOR > 0:
            rand_no = np.random.rand()
            if rand_no < self.options.CROP_PROB:
                center, scale = random_crop(center, scale,
                                            crop_scale_factor=1 - self.options.CROP_FACTOR,
                                            axis='y')

        imgname = os.path.join(self.img_dir, self.imgname[index])
        try:
            cv_img = read_img(imgname)
        except Exception as E:
            logger.info(f'@{imgname} from {self.dataset}')

        if self.is_train and 'closeup' in self.dataset:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)

        # Because closeup images are stored in rotated format
        if (self.dataset == 'bedlam-val' or self.dataset == 'bedlam-test') and self.width[index] == 720:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)

        orig_shape = np.array(cv_img.shape)[:2]

        pose = self.pose_cam[index].copy()
        keypoints = self.j2d_processing(keypoints, center, sc * scale)
        if self.options.proj_verts:
            proj_verts = self.j2d_processing(proj_verts, center, sc * scale)
            item['proj_verts'] = torch.from_numpy(proj_verts).float()
        # Process image
        try:
            img = self.rgb_processing(cv_img, center, sc*scale, kp2d=keypoints,
                                      img_res=self.options.IMG_RES)
        except Exception as E:
            logger.info(f'@{imgname} from {self.dataset}')
            print(E)

        img = torch.from_numpy(img).float()

        # if self.is_train or ((not self.is_train) and 'bedlam' in self.dataset):
        center_l = self.center_l[index]
        center_r = self.center_r[index]
        scale_l = self.scale_l[index]
        scale_r = self.scale_r[index]
        try:
            lhand_img = self.rgb_processing(cv_img, center_l, sc * scale_l, img_res=self.options.IMG_RES)
            rhand_img = self.rgb_processing(cv_img, center_r, sc * scale_r, img_res=self.options.IMG_RES)
        except Exception as E:
            logger.info(f'@{imgname} from {self.dataset}')

        item['lhand_img'] = lhand_img
        item['rhand_img'] = rhand_img
        item['lhand_pose'] = torch.from_numpy(self.lhand_pose[index]).float()
        item['rhand_pose'] = torch.from_numpy(self.rhand_pose[index]).float()

        #Debug 
        # if lhand_img is not None:
        #     save_img = lhand_img.transpose(1,2,0)*255
        #     save_img = np.clip(save_img, 0 , 255).astype(np.uint8)
        #     cv2.imwrite('temp'+str(index)+'.png', save_img)

        item['img'] = self.normalize_img(img)

        item['pose'] = torch.from_numpy(pose).float()
        item['betas'] = torch.from_numpy(self.betas[index]).float()
        item['imgname'] = imgname

        if 'cam_int' in self.data.files:
            item['focal_length'] = torch.tensor([self.cam_int[index][0, 0], self.cam_int[index][1, 1]])
        if 'cam_ext' in self.data.files:
            item['translation'] = self.cam_ext[index][:, 3]
        if 'trans_cam' in self.data.files:
            item['translation'][:3] += self.trans_cam[index]

        item['keypoints_orig'] = torch.from_numpy(keypoints_orig).float()
        item['keypoints'] = torch.from_numpy(keypoints).float()
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset

        if not self.is_train:
            if '3dpw' in self.dataset:
                if self.gender[index] == 1:
                    gt_smpl_out = self.smpl_female(
                    global_orient=item['pose'].unsqueeze(0)[:, :3],
                    body_pose=item['pose'].unsqueeze(0)[:, 3:],
                    betas=item['betas'].unsqueeze(0),)
                    gt_vertices = gt_smpl_out.vertices
                else:
                    gt_smpl_out = self.smpl_male(
                        global_orient=item['pose'].unsqueeze(0)[:, :3],
                        body_pose=item['pose'].unsqueeze(0)[:, 3:],
                        betas=item['betas'].unsqueeze(0),
                    )
                    gt_vertices = gt_smpl_out.vertices

                item['vertices'] = gt_vertices[0].float()
            else:
                item['vertices'] = torch.zeros((6890, 3)).float()

        if not self.is_train:
            item['dataset_index'] = self.options.VAL_DS.split('_').index(self.dataset)
            item['img_full'] = np.transpose(cv_img.astype('float32'),(2,0,1))/255.0
        return item

    def __len__(self):
        if self.is_train and 'agora' not in self.dataset and '3dpw' not in self.dataset:
            return int(self.options.CROP_PERCENT * len(self.imgname))
        else:
            return len(self.imgname)