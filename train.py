import argparse
import math
import time

import numpy as np
import cv2
import os
import glob
import random
import paddle
from paddle.nn import functional as F


from data_loader.dataset_face import GFPGAN_degradation
from data_loader.dataset_face import FaceDataset
from model.gpen import GPEN as FullGenerator
from model.discriminator_styleganv2 import StyleGANv2Discriminator as Discriminator
from loss.id_loss import IDLoss
from metrics.psnr import psnr
import metrics.fid as fid

import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = paddle.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape([grad_real.shape[0], -1]).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred, loss_funcs=None, fake_img=None, real_img=None, input_img=None):
    smooth_l1_loss, id_loss = loss_funcs

    loss = F.softplus(-fake_pred).mean()
    loss_l1 = smooth_l1_loss(fake_img, real_img)
    loss_id, __, __ = id_loss(fake_img, real_img, input_img)
    loss += 1.0 * loss_l1 + 1.0 * loss_id

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = paddle.randn(fake_img.shape) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = paddle.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = paddle.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def stop_grad(model, flag=True):
    for p in model.parameters():
        p.stop_gradient = flag
    return model

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.state_dict())
    par2 = dict(model2.state_dict())
    for k in par1.keys():
        par1[k] = par1[k]*decay + par2[k] * (1-decay)
    model1.load_dict(par1)

def validation(model, degrader, FID_model, size, path, amount=1000):
    hq_files = sorted(glob.glob(os.path.join(path, '*.*')))[:amount]

    fid_sum = 0
    psnr_sum = 0
    model.eval()
    i = 0
    print(f'\ntesting start:', end='')
    for hq_f in hq_files:
        img_gt = cv2.imread(hq_f, cv2.IMREAD_COLOR)
        img_gt = cv2.resize(img_gt, (size, size), interpolation=cv2.INTER_NEAREST)
        img_gt = img_gt.astype(np.float32) / 255.
        img_gt, img_lq = degrader.degrade_process(img_gt)

        img_gt = (img_gt - 0.5) / 0.5
        img_lq = (img_lq - 0.5) / 0.5

        img_gt = paddle.to_tensor(np.transpose(img_gt, (2, 0, 1))).unsqueeze(0)
        img_lq = paddle.to_tensor(np.transpose(img_lq, (2, 0, 1))).unsqueeze(0)

        #         img_gt = F.interpolate(img_gt, (size, size))
        #         img_lq = F.interpolate(img_lq, (size, size))

        img_gt = paddle.flip(img_gt, [1])
        img_lq = paddle.flip(img_lq, [1])

        with paddle.no_grad():
            img_out, __ = model(img_lq)

            psnr_sum += psnr(img_gt, img_out)
            FID_model.update(img_out, img_gt)
            fid_score = FID_model.accumulate()
            fid_sum += fid_score
        i += 1

        if (i - 1) % 100 == 0:
            print(f'\ntesting: {i}/{len(hq_files)}', end='')
        if i % 10 == 0:
            print('.', end='')
        if i == 1000:
            print(f'\ntest:  fid:{fid_sum / 1000}')

    return psnr_sum / len(hq_files), fid_sum / len(hq_files)


def train(loader, generator, discriminator, losses, g_optim, d_optim, g_ema, args):

    mean_path_length = 0
    loss_dict = {}
    accum = 0.5 ** (32 / (10 * 1000))

    i = args.start_iter - 1
    start_time = time.time()
    for epoch in range(1000000):
        if i > args.max_iter:
            print('Done!')
            break
        for idx, (degraded_img, real_img) in enumerate(loader):
            i += 1
            if i > args.max_iter:
                print('Done!')
                break

            degraded_img = paddle.to_tensor(degraded_img)
            real_img = paddle.to_tensor(real_img)

            stop_grad(generator, True)
            stop_grad(discriminator, False)

            fake_img, _ = generator(degraded_img)
            fake_pred = discriminator(fake_img)
            real_pred = discriminator(real_img)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            loss_dict['d'] = d_loss
            loss_dict['real_score'] = real_pred.mean()
            loss_dict['fake_score'] = fake_pred.mean()

            d_optim.clear_grad()
            d_loss.backward()

            d_optim.step()

            d_regularize = i % args.d_reg_every == 0
            if d_regularize:
                real_img.stop_gradient=False
                real_pred = discriminator(real_img)
                r1_loss = d_r1_loss(real_pred, real_img)
                g_optim.clear_grad()
                (10 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

                d_optim.step()

            # loss_dict['r1'] = r1_loss

            stop_grad(generator, False)
            stop_grad(discriminator, True)

            fake_img, _ = generator(degraded_img)
            fake_pred = discriminator(fake_img)
            g_loss = F.softplus(-fake_pred).mean()
            smooth_l1_loss, id_loss = losses
            g_loss += smooth_l1_loss(fake_img, real_img)
            g_loss += 0.02 * id_loss(fake_img, real_img, degraded_img)[0]

            loss_dict['g'] = g_loss

            g_optim.clear_grad()
            g_loss.backward()
            g_optim.step()


            current_lr_g = g_optim.get_lr()
            current_lr_d = d_optim.get_lr()

            g_regularize = i % args.g_reg_every == 0
            if g_regularize:
                fake_img, latents = generator(degraded_img, return_latents=True)
                path_loss, mean_path_length, path_lengths = g_path_regularize(
                    fake_img, latents, mean_path_length
                )
                g_optim.clear_grad()
                weighted_path_loss = 2 * args.g_reg_every * path_loss

                weighted_path_loss.backward()

                g_optim.step()

            accumulate(g_ema, generator, accum)  # g_ema =

            d_loss_val = loss_dict['d'].mean().item()
            g_loss_val = loss_dict['g'].mean().item()
            # r1_val = loss_dict['r1'].mean().item()
            print(
                f'\rstep: {i}/{args.max_iter}; d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; lr: {current_lr_g:.7f}/{current_lr_d:.7f} ; time_used: {(time.time() - start_time) / 60 :.1f}min',
                end='', flush=True)

            if i % args.save_iter == 0:
                with paddle.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema(degraded_img)
                    degraded_img = degraded_img.transpose([0, 2, 3, 1])
                    real_img = real_img.transpose([0, 2, 3, 1])
                    sample = sample.transpose([0, 2, 3, 1])

                    for j in range(sample.shape[0]):
                        mid_sample = paddle.concat((degraded_img[j], sample[j], real_img[j]), 0)
                        if j == 0:
                            sample_result = mid_sample
                        else:
                            sample_result = paddle.concat((sample_result, mid_sample), 1)
                    sample = sample_result
                    sample = cv2.cvtColor((sample.numpy() + 1) / 2 * 255, cv2.COLOR_RGB2BGR)
                    file_name = args.sample_dir + str(i).zfill(6) + '_g_ema.png'
                    cv2.imwrite(file_name, sample)

                if i and i % args.save_iter*2 == 0:
                    psnr_value, FID_value = validation(g_ema, degrader_model, FID_MODEL, args.size, args.test_path,
                                                        100)
                    logger.info(
                        'step:[{}/{}]\t g_ema:\t psnr={:.5f}\t FID={:.3f}'.format(i, args.max_iter, psnr_value, FID_value))
            if i and i % args.save_iter*4 == 0:  # save_freq
                file_name = args.ckpt_dir + str(i).zfill(6) + '.pdparams'
                paddle.save(
                    {
                        'g': generator.state_dict(),
                        'd': discriminator.state_dict(),
                        'g_ema': g_ema.state_dict(),
                    },
                    file_name,
                )

            if idx == len(loader) - 1:
                break

def parse_args():
    parser = argparse.ArgumentParser(description='Model train.')
    parser.add_argument('--train_path', type=str, default='../data/train/')
    parser.add_argument('--test_path', type=str, default='../data/test_imgs/')
    parser.add_argument('--size', dest='size', default=256, type=int, help='the size of input and output images')
    parser.add_argument('--mul', dest='mul', default=1, type=int)
    parser.add_argument('--narrow', dest='narrow', default=0.5, type=float)
    parser.add_argument('--is_concat', dest='is_concat', default=True, type=bool)
    parser.add_argument('--pretrain', type=str, default=None)

    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=150000)
    parser.add_argument('--save_iter', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)

    parser.add_argument('--ckpt_dir', type=str, default='ckpts/')
    parser.add_argument('--sample_dir', type=str, default='samples/')

    return parser.parse_args()


if __name__ == '__main__':


    args = parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)


    generator = FullGenerator(args.size, 512, 8, channel_multiplier=args.mul,narrow=args.narrow, is_concat=args.is_concat)
    discriminator = Discriminator(args.size)
    g_ema = FullGenerator(args.size, 512, 8, channel_multiplier=args.mul,narrow=args.narrow, is_concat=args.is_concat)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    if args.pretrain is not None:
        print('load model:', args.pretrain)

        ckpt = paddle.load(args.pretrain)

        generator.load_dict(ckpt['g'])
        discriminator.load_dict(ckpt['d'])
        g_ema.load_dict(ckpt['g_ema'])

    train_dataset = FaceDataset(args.train_path, args.size)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    smooth_l1_loss = paddle.nn.SmoothL1Loss()
    id_loss = IDLoss()

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = paddle.optimizer.Adam(
        beta1=0 ** g_reg_ratio,
        beta2=0.99 ** g_reg_ratio,
        learning_rate=args.lr * g_reg_ratio,
        parameters=generator.parameters()
    )

    d_optim = paddle.optimizer.Adam(
        learning_rate=args.lr * d_reg_ratio,
        beta1=0 ** d_reg_ratio,
        beta2=0.99 ** d_reg_ratio,
        parameters=discriminator.parameters()
    )
    g_optim.clear_grad()
    d_optim.clear_grad()

    degrader_model = GFPGAN_degradation()
    FID_MODEL = fid.FID(use_GPU=True)

    #paddle.save(g_ema.state_dict(), args.ckpt_dir + "/G_256_repo_test.pdparams")
    logger = get_logger(args.ckpt_dir + '/train.log')
    logger.info('start training!')

    #train(train_loader, generator, discriminator, [smooth_l1_loss, id_loss],g_optim , d_optim, g_ema, args)

    paddle.save(g_ema.state_dict(), args.ckpt_dir + "/G_256_repo_final.pdparams")
    logger.info('finish training!')