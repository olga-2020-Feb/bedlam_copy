import itertools
from yacs.config import CfgNode as CN
from typing import List, Union
from flatten_dict import flatten, unflatten


SMPL_MODEL_DIR = r'../storage/data/body_models/SMPL_python_v.1.1.0/smpl/models'
SMPLX_MODEL_DIR = r'../storage/data/body_models/smplx/models/smplx'
MANO_MODEL_DIR = r'../storage/data/body_models/mano/mano_v1_2/models/'

JOINT_REGRESSOR_TRAIN_EXTRA = r'../storage/data/utils/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = r'../storage/data/utils/J_regressor_h36m.npy'
SMPL_MEAN_PARAMS = r'../storage/data/utils/smpl_mean_params.npz'
JOINT_REGRESSOR_14 = r'../storage/data/utils/SMPLX_to_J14.pkl'
SMPLX2SMPL = r'../storage/data/utils/smplx2smpl.pkl'
MEAN_PARAMS = r'../storage/data/utils/all_means.pkl'
DOWNSAMPLE_MAT_SMPLX_PATH = r'../storage/data/utils/downsample_mat_smplx.pkl'

DATASET_FOLDERS = {
    'orbit-archviz-15': r'../storage/data/bedlam/training_images/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png',
    'static-hdri': r'../storage/data/bedlam/training_images/20221010_3_1000_batch01hand_6fps/png',
    'static-hdri-bmi': r'../storage/data/bedlam/training_images/20221019_3_250_highbmihand_6fps/png',

}

DATASET_FILES = [
    {
        'orbit-archviz-15': r'../storage/data/bedlam/training_labels/all_npz_12_training/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps.npz',
        'static-hdri': r'../storage/data/bedlam/training_labels/all_npz_12_training/20221010_3_1000_batch01hand_6fps.npz',
        'static-hdri-bmi': r'../storage/data/bedlam/training_labels/all_npz_12_training/20221019_3_250_highbmihand_6fps.npz',
    },
    {
        'orbit-archviz-15': r'../storage/data/bedlam/training_labels/all_npz_12_training/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps.npz',
        'static-hdri': r'../storage/data/bedlam/training_labels/all_npz_12_training/20221010_3_1000_batch01hand_6fps.npz',
        'static-hdri-bmi': r'../storage/data/bedlam/training_labels/all_npz_12_training/20221019_3_250_highbmihand_6fps.npz',
    }
]

# Download the models from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch and update the path
PRETRAINED_CKPT_FOLDER = {
    'hrnet_w32-coco': r'../storage/data/ckpt/pretrained/pose_hrnet_w32_256x192.pth',
    'hrnet_w32-imagenet': r'../storage/data/ckpt/pretrained/hrnetv2_w32_imagenet_pretrained.pth',
    'hrnet_w32-scratch': '',
    'hrnet_w48-coco': r'../storage/data/ckpt/pretrained/pose_hrnet_w48_256x192.pth',
    'hrnet_w48-imagenet': r'../storage/data/ckpt/pretrained/hrnetv2_w48_imagenet_pretrained.pth',
    'hrnet_w48-scratch': '',

}
hparams = CN()
# General settings
hparams.LOG_DIR = '../storage/logs'
hparams.EXP_NAME = 'default'
hparams.SEED_VALUE = -1
hparams.RUN_TEST = False

# Dataset hparams
hparams.DATASET = CN()
hparams.DATASET.NOISE_FACTOR = 0.4
hparams.DATASET.SCALE_FACTOR = 0.25
hparams.DATASET.CROP_PROB = 0.0
hparams.DATASET.CROP_FACTOR = 0.0
hparams.DATASET.BATCH_SIZE = 64
hparams.DATASET.NUM_WORKERS = 8
hparams.DATASET.PIN_MEMORY = True
hparams.DATASET.SHUFFLE_TRAIN = True
hparams.DATASET.TRAIN_DS = 'all'
hparams.DATASET.VAL_DS = '3dpw-val-cam'
hparams.DATASET.IMG_RES = 224
hparams.DATASET.MESH_COLOR = 'pinkish'
hparams.DATASET.DATASETS_AND_RATIOS = 'agora'
hparams.DATASET.CROP_PERCENT = 1.0
hparams.DATASET.ALB = False
hparams.DATASET.ALB_PROB = 0.3
hparams.DATASET.proj_verts = False
hparams.DATASET.FOCAL_LENGTH = 5000
hparams.DATASET.PARTS_COUNT = 0
hparams.DATASET.PART_IDX = 0

# optimizer config
hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.TYPE = 'adam'
hparams.OPTIMIZER.LR = 5e-5 #0.0001 # 0.00003 
hparams.OPTIMIZER.WD = 0.0

# Training process hparams
hparams.TRAINING = CN()
hparams.TRAINING.RESUME = None
hparams.TRAINING.PRETRAINED_CKPT = None
hparams.TRAINING.PRETRAINED_LIT = None
hparams.TRAINING.MAX_EPOCHS = 100
hparams.TRAINING.LOG_SAVE_INTERVAL = 50
hparams.TRAINING.LOG_FREQ_TB_IMAGES = 500
hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH = 1
hparams.TRAINING.RELOAD_DATALOADERS_EVERY_EPOCH = True
hparams.TRAINING.TEST_BEFORE_TRAINING = False
hparams.TRAINING.SAVE_IMAGES = False
hparams.TRAINING.USE_AMP = False
hparams.TRAINING.GT_VIS = False
hparams.TRAINING.WP_VIS = False
hparams.TRAINING.FP_VIS = False
hparams.TRAINING.MESH_VIS = False
hparams.TRAINING.fullbody_mode = 'default'
hparams.TRAINING.ckpt_mode = ''
hparams.TRAINING.reduce_lr = False
hparams.TRAINING.increase_lr = False
hparams.TRAINING.hand_ckpt = ''
hparams.TRAINING.body_ckpt = ''
hparams.TRAINING.finetune = False
hparams.TRAINING.LOG_EVERY_N_STEPS = 50
hparams.TRAINING.CKPT_PREF = ''
hparams.TRAINING.CKPT_POSTFIX = ''

# Training process hparams
hparams.TESTING = CN()
hparams.TESTING.GT_VIS = False
hparams.TESTING.WP_VIS = False
hparams.TESTING.FP_VIS = False
hparams.TESTING.MESH_VIS = False

# MODEL  hparams
hparams.MODEL = CN()
hparams.MODEL.BACKBONE = 'resnet50'
hparams.MODEL.SHAPE_LOSS_WEIGHT = 0.
hparams.MODEL.JOINT_LOSS_WEIGHT = 5.
hparams.MODEL.KEYPOINT_LOSS_WEIGHT = 10.
hparams.MODEL.POSE_LOSS_WEIGHT = 1.
hparams.MODEL.BETA_LOSS_WEIGHT = 0.001
hparams.MODEL.LOSS_WEIGHT = 60.

hparams.TRIAL = CN()
hparams.TRIAL.visualize = False
hparams.TRIAL.bedlam_bbox = False
hparams.TRIAL.verts_loss = False
hparams.TRIAL.only_verts_loss = False
hparams.TRIAL.keypoints_loss = False
hparams.TRIAL.only_keypoints_loss = False
hparams.TRIAL.param_loss = False
hparams.TRIAL.losses_abl = 'None'
hparams.TRIAL.criterion = 'mse'
hparams.TRIAL.version = 'synthetic'
hparams.TRIAL.finetune_3dpw = False


def get_hparams_defaults():
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()


def update_hparams(hparams_file):
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()


def update_hparams_from_dict(cfg_dict):
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()


def get_grid_search_configs(config, excluded_keys=[]):

    def bool_to_string(x: Union[List[bool], bool]) -> Union[List[str], str]:
        """
        boolean to string conversion
        :param x: list or bool to be converted
        :return: string converted thinghat
        """
        if isinstance(x, bool):
            return [str(x)]
        for i, j in enumerate(x):
            x[i] = str(j)
        return x

    # exclude from grid search

    flattened_config_dict = flatten(config, reducer='path')
    hyper_params = []

    for k, v in flattened_config_dict.items():
        if isinstance(v, list):
            if k in excluded_keys:
                flattened_config_dict[k] = ['+'.join(v)]
            elif len(v) > 1:
                hyper_params += [k]

        if isinstance(v, list) and isinstance(v[0], bool) :
            flattened_config_dict[k] = bool_to_string(v)

        if not isinstance(v,list):
            if isinstance(v, bool):
                flattened_config_dict[k] = bool_to_string(v)
            else:
                flattened_config_dict[k] = [v]

    keys, values = zip(*flattened_config_dict.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for exp_id, exp in enumerate(experiments):
        for param in excluded_keys:
            exp[param] = exp[param].strip().split('+')
        for param_name, param_value in exp.items():
            # print(param_name,type(param_value))
            if isinstance(param_value, list) and (param_value[0] in ['True', 'False']):
                exp[param_name] = [True if x == 'True' else False for x in param_value]
            if param_value in ['True', 'False']:
                if param_value == 'True':
                    exp[param_name] = True
                else:
                    exp[param_name] = False

        experiments[exp_id] = unflatten(exp, splitter='path')

    return experiments, hyper_params
