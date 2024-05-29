import os
import shutil
import sys
import torch
import time
import yaml
import argparse
import numpy as np
import pytorch_lightning as pl
from loguru import logger
from flatten_dict import flatten, unflatten
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import ProgressBar
from train.utils.os_utils import copy_code
from train.core.hmr_trainer import HMRTrainer, HMRPartTrainer
from train.utils.train_utils import set_seed, update_hparams
#from pytorch_lightning.loggers import WandbLogger

sys.path.append('.')

def get_checkpoint_name(hparams):
    return f'{hparams.DATASET.PART_IDX}_of_{hparams.DATASET.PARTS_COUNT}_{hparams.TRAINING.CKPT_POSTFIX}.pth.tar'

def train(hparams, fast_dev_run=False):
    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(hparams.SEED_VALUE)
    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    # copy the code for reproducibility
    # copy_code(
    #     output_folder=log_dir,
    #     curr_folder=os.path.dirname(os.path.abspath(__file__))
    # )

    logger.info(torch.cuda.get_device_properties(device))
    logger.info(f'Hyperparameters: \n {hparams}')
    experiment_loggers = []
    # initialize tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        log_graph=False,
    )
    experiment_loggers.append(tb_logger)

    model = HMRPartTrainer(hparams=hparams).to(device)

    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        verbose=True,
        save_top_k=5,
        mode='min',
        filename = f'{hparams.DATASET.PART_IDX}_of_{hparams.DATASET.PARTS_COUNT}_{hparams.TRAINING.CKPT_PREF}'
    )
    log_every_n_steps = 50
    if hparams.TRAINING.get('LOG_EVERY_N_STEPS') is not None:
        log_every_n_steps = hparams.TRAINING.LOG_EVERY_N_STEPS
    trainer = pl.Trainer(
        gpus=1,
        logger=experiment_loggers,
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        callbacks=[ckpt_callback, ProgressBar(refresh_rate=200)],
        default_root_dir=log_dir,
        check_val_every_n_epoch=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev_run,
        resume_from_checkpoint=hparams.TRAINING.RESUME,
        log_every_n_steps= log_every_n_steps,
        enable_checkpointing=True,
    )

    logger.info('*** Started training ***')

    if args.test:
        trainer.test(model, ckpt_path=hparams.TRAINING.RESUME)
    else:
        trainer.fit(model, ckpt_path=hparams.TRAINING.RESUME)

    torch.save(model.model.head.state_dict(),
               os.path.join(hparams.LOG_DIR, get_checkpoint_name(hparams)))


def main(part_idx = None):
    global args, hparams, hparams
    parser = prepare_argparser()
    args = parser.parse_args()
    logger.info(f'Input arguments: \n {args}')
    torch.cuda.empty_cache()
    # Update hparams with config file and from args
    hparams = update_hparams(args.cfg)
    if part_idx is not None:
        hparams.DATASET.PART_IDX = part_idx
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')

    logdir = os.path.join(args.log_dir, hparams.EXP_NAME + logtime)
    os.makedirs(logdir, exist_ok=True)
    orig_log_dir = hparams.LOG_DIR
    hparams.LOG_DIR = logdir
    if args.ckpt:
        hparams.TRAINING.RESUME = args.ckpt
    # Load the last checkpoint using the epoch id.
    if args.resume and hparams.TRAINING.RESUME is None:
        ckpt_files = []
        for root, dirs, files in os.walk(args.log_dir, topdown=False):
            for f in files:
                if f.endswith('.ckpt'):
                    ckpt_files.append(os.path.join(root, f))

        epoch_idx = [int(x.split('=')[-1].split('.')[0]) for x in ckpt_files]
        if len(epoch_idx) == 0:
            ckpt_file = None
        else:
            last_epoch_idx = np.argsort(epoch_idx)[-1]
            ckpt_file = ckpt_files[last_epoch_idx]
        logger.info('Loading CKPT', ckpt_file)
        hparams.TRAINING.RESUME = ckpt_file
    if args.test:
        hparams.RUN_TEST = True

    def save_dict_to_yaml(obj, filename, mode='w'):
        with open(filename, mode) as f:
            yaml.dump(obj, f, default_flow_style=False)

    # save final config
    save_dict_to_yaml(
        unflatten(flatten(hparams)),
        os.path.join(hparams.LOG_DIR, 'config_to_run.yaml')
    )
    train(hparams, fast_dev_run=args.fdr)


def prepare_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--log_dir', type=str, help='log dir path', default='./logs')
    parser.add_argument('--fdr', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--ckpt', type=str)
    return parser


def collect_checkpoints():
    global args, hparams
    log_dir = os.path.abspath( r'logs')
    log_folders = os.listdir(log_dir)
    new_folder_name = 'first 10 small 128 n_iter = 1'
    dst_folder = os.path.join(log_dir, new_folder_name)
    os.makedirs(dst_folder, exist_ok=True)
    parser = prepare_argparser()
    args = parser.parse_args()
    logger.info(f'Input arguments: \n {args}')
    torch.cuda.empty_cache()
    # Update hparams with config file and from args
    hparams = update_hparams(args.cfg)
    for log_folder in log_folders:
        if log_folder != new_folder_name:
            src_folder = os.path.join(log_dir, log_folder)
            if os.path.exists(src_folder):
                srs_files = os.listdir(src_folder)
                for srs_file in srs_files:
                    if f'{hparams.TRAINING.CKPT_POSTFIX}.pth.tar' in srs_file:
                        src_file_path = os.path.join(src_folder, srs_file)
                        dst_file_path = os.path.join(dst_folder, srs_file)
                        shutil.copyfile(src_file_path, dst_file_path)


if __name__ == '__main__':
    collect_checkpoints()

    for part_idx in range(5):
        main(part_idx)

