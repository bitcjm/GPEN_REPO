import argparse
import random

import numpy as np
import cv2
import os
import glob
import paddle


from data_loader.dataset_face import GFPGAN_degradation
from model.gpen import GPEN as FullGenerator
from metrics.psnr import psnr
import metrics.fid as fid



def validation(model, degrader, FID_model, size, path):
    hq_files = sorted(glob.glob(os.path.join(path, '*.*')))

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


def parse_args():
    parser = argparse.ArgumentParser(description='Model test.')
    parser.add_argument('--test_path', type=str, default='../data/test_imgs/')

    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--w', dest='gen_model_path', help='The path of pretrain generator model',
                        type=str, default='data/GPEN/G_256_weight_best.pdparams')

    parser.add_argument('--size', dest='size', default=256, type=int, help='the size of input and output images')
    parser.add_argument('--mul', dest='mul', default=1, type=int)
    parser.add_argument('--narrow', dest='narrow', default=0.5, type=float)
    parser.add_argument('--is_concat', dest='is_concat', default=True, type=bool)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    paddle.seed(100)
    np.random.seed(100)
    random.seed(100)
    generator = FullGenerator(args.size, 512, 8, channel_multiplier=args.mul,narrow=args.narrow, is_concat=args.is_concat)

    if args.pretrain == True:
        print('load model:\n\tgenertor:', args.gen_model_path,)

        ckpt_g = paddle.load(args.gen_model_path)
        generator.load_dict(ckpt_g)

    degrader_model = GFPGAN_degradation()
    FID_MODEL = fid.FID(use_GPU=True)

    psnr_value, FID_value = validation(generator, degrader_model, FID_MODEL, args.size, args.test_path)
    print(f'\npsnr_avg: {psnr_value}, FID_avg: {FID_value}')



