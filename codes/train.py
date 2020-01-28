import os.path
import sys
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging

import torch

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from models.modules.LPIPS import compute_dists as lpips

def main():
    # Load options YAML file
    parser = argparse.ArgumentParser(description="BasicSR")
    parser.add_argument("-o", "-opt", type=str, required=True, help="Path to training YAML options file.")
    opt = option.iterable_missing_hook(
        option.parse(parser.parse_args().opt, is_train=True)
    )

    # Configure logging
    util.setup_logger(None, opt["path"]["log"], "train", level=logging.INFO, screen=True)
    util.setup_logger("val", opt["path"]["log"], "val", level=logging.INFO)
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    # check if a resume state is available
    if opt["path"]["resume_state"]:
        resume_state_path = opt["path"]["resume_state"]
        if os.path.isdir(resume_state_path):
            import glob
            resume_state_path = util.sorted_nicely(
                glob.glob(os.path.join(os.path.normpath(resume_state_path), "*.state"))
            )[-1]
        resume_state = torch.load(resume_state_path)
        logger.info(f"Set [resume_state] to \"{resume_state_path}\"")
        logger.info("Resuming training from epoch: {}, iter: {}".format(
            resume_state["epoch"], resume_state["iter"]
        ))
        option.check_resume(opt)
    else:
        # start a new train state
        resume_state = None
        util.mkdir_and_rename(opt["path"]["experiments_root"])  # rename old folder if exists
        util.mkdirs((
            path for key, path in opt["path"].items() if (
                key != "experiments_root" and
                "pretrain_model" not in key and
                "resume" not in key
            )
        ))
        current_step = 0
        epoch = 0
        logger.info('Start training from epoch: {:d}, iter: {:d}'.format(epoch, current_step))

    # Create tensorboard log
    if opt["use_tb_logger"] and "debug" not in opt["name"]:
        from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(f"../tb_logger/{opt['name']}")

    # Generate a pseudo-random seed
    seed = opt["train"]["manual_seed"]
    if seed is None or seed <= -1:
        seed = random.randint(0, (2**32)-1)  # must be between 0 and (2^32)-1
    logger.info(f"Selected seed: {seed}")
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloader
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            logger.info("Number of train images: {:,d}, iters: {:,d}".format(
                len(train_set), train_size
            ))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info("Total epochs needed: {:d} for iters {:,d}".format(
                total_epochs, total_iters
            ))
            train_loader = create_dataloader(
                train_set,
                phase,
                batch_size=dataset_opt["batch_size"],
                shuffle=dataset_opt["use_shuffle"],
                num_workers=dataset_opt["n_workers"]
            )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                phase,
                batch_size=dataset_opt["batch_size"],
                shuffle=dataset_opt["use_shuffle"],
                num_workers=dataset_opt["n_workers"]
            )
            logger.info("Number of val images in [{:s}]: {:d}".format(
                dataset_opt["name"], len(val_set)
            ))
        else:
            raise NotImplementedError(f"Phase [{phase}] is not recognized.")
    assert train_loader is not None

    # create model
    model = create_model(opt)

    # circumvent pytorch warning
    # we pass in the iteration count to scheduler.step(), so the warning doesn't apply
    for scheduler in model.schedulers:
        if hasattr(scheduler, "_step_count"):
            scheduler._step_count = 0

    # resume training
    if resume_state:
        epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
        model.update_schedulers(opt["train"]) # updated schedulers in case YAML configuration has changed

    while current_step <= total_iters:
        for n, train_data in enumerate(train_loader, start=1):
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_step-1)
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            # log
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}:{: .4e} ".format(k, v)
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)
            # save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                model.save(current_step)
                model.save_training_state(epoch + (n >= len(train_loader)), current_step)
                logger.info("Models and training states saved.")
            # validation
            if current_step % opt["train"]["val_freq"] == 0:
                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_lpips = 0.0
                idx = 0
                val_gt = []
                val_sr = []
                for val_data in val_loader:
                    idx += 1

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    
                    if opt["datasets"]["train"]["znorm"]:
                        # If the image range is [-1,1]
                        sr_img = util.tensor2img(visuals["SR"], min_max=(-1, 1))  # uint8
                        gt_img = util.tensor2img(visuals["HR"], min_max=(-1, 1))  # uint8
                    else:
                        # Image range is [0,1]
                        sr_img = util.tensor2img(visuals["SR"])  # uint8
                        gt_img = util.tensor2img(visuals["HR"])  # uint8
                    
                    if "debug" in opt["name"]:
                        print(f"SR value (min / max): {sr_img.min()} / {sr_img.max()}")
                        print(f"GT value (min / max): {gt_img.min()} / {gt_img.max()}")
                    
                    # Save SR images for reference
                    img_name, _ = os.path.splitext(os.path.basename(val_data["LR_path"][0]))
                    img_dir = os.path.join(opt["path"]["val_images"], img_name)
                    util.mkdir(img_dir)
                    util.save_img(
                        sr_img,
                        os.path.join(img_dir, f"{img_name}_{current_step}.png")
                    )

                    # calculate PSNR, SSIM and LPIPS distance
                    crop_size = opt["scale"]
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    if gt_img.ndim == 2:
                        # one channel (grayscale)
                        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]
                    else:
                        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    if sr_img.ndim == 2:
                        # one channel (grayscale)
                        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
                    else:
                        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    val_gt.append(cropped_gt_img)
                    val_sr.append(cropped_sr_img)
                    
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                    avg_ssim += util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)

                # calculate the average of the metric values
                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                # calculate lpips on all given images
                # todo ; LPIPS only works for RGB images, add a check if the image is RGB
                avg_lpips = lpips.calculate_lpips(val_sr, val_gt)

                # log
                logger.info("# Validation # PSNR: {:.5g}, SSIM: {:.5g}, LPIPS: {:.5g}".format(
                    avg_psnr, avg_ssim, avg_lpips
                ))
                logger_val = logging.getLogger("val")
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.5g}, ssim: {:.5g}, lpips: {:.5g}'.format(
                    epoch, current_step, avg_psnr, avg_ssim, avg_lpips
                ))
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)
                    tb_logger.add_scalar("ssim", avg_ssim, current_step)
                    tb_logger.add_scalar("lpips", avg_lpips, current_step)
        epoch += 1

    logger.info("Saving the final model.")
    model.save("latest")
    logger.info("End of training.")


if __name__ == '__main__':
    main()
