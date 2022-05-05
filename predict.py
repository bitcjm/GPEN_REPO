import argparse
import random

import cv2
import numpy as np
import paddle

from model.gpen import GPEN as MainModel
from data_loader.dataset_face import GFPGAN_degradation
import metrics.fid as fid
from metrics.psnr import psnr

import warnings
warnings.filterwarnings('ignore')

def data_loader(path, size=256):
    degrader = GFPGAN_degradation()

    img_gt = cv2.imread(path, cv2.IMREAD_COLOR)

    img_gt = cv2.resize(img_gt, (size, size), interpolation=cv2.INTER_NEAREST)

    img_gt = img_gt.astype(np.float32) / 255.
    img_gt, img_lq = degrader.degrade_process(img_gt)

    img_gt = (paddle.to_tensor(img_gt) - 0.5) / 0.5
    img_lq = (paddle.to_tensor(img_lq) - 0.5) / 0.5

    img_gt = img_gt.transpose([2, 0, 1]).flip(0).unsqueeze(0)
    img_lq = img_lq.transpose([2, 0, 1]).flip(0).unsqueeze(0)

    return np.array(img_lq).astype('float32'), np.array(img_gt).astype('float32')

def parse_args():
    parser = argparse.ArgumentParser(description='Model predict')

    parser.add_argument('--gpus', dest='gpus', default='0', type=str)
    parser.add_argument('--img', dest='img', default='data/GPEN/predict/test_img.png', type=str)
    parser.add_argument('--w', dest='model_weight',default='data/GPEN/G_256_weight_best.pdparams', type=str)
    parser.add_argument('--size', dest='size', default=256, type=int)
    parser.add_argument('--mul', dest='mul', default=1, type=int)
    parser.add_argument('--narrow', dest='narrow', default=0.5, type=float)
    parser.add_argument('--is_concat', dest='is_concat', default=True, type=bool)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    paddle.device.set_device(f'gpu:{args.gpus}')

    paddle.seed(100)
    np.random.seed(100)
    random.seed(100)

    # 数据集加载
    input_array,target_array = data_loader(args.img, args.size)
    input_tensor = paddle.to_tensor(input_array)
    target_tensor = paddle.to_tensor(target_array)

    model = MainModel(args.size, 512, 8, channel_multiplier=args.mul,narrow=args.narrow, is_concat=args.is_concat)
    model.load_dict(paddle.load(args.model_weight))
    model.eval()

    FID_model = fid.FID(use_GPU=True)

    with paddle.no_grad():
        output,_ = model(input_tensor)
        psnr_score = psnr(target_tensor, output)
        FID_model.update(output, target_tensor)
        fid_score = FID_model.accumulate()

    input_tensor = input_tensor.transpose([0, 2, 3, 1])
    target_tensor = target_tensor.transpose([0, 2, 3, 1])
    output = output.transpose([0, 2, 3, 1])
    sample_result = paddle.concat((input_tensor[0], output[0], target_tensor[0]), 1)
    sample = cv2.cvtColor((sample_result.numpy() + 1) / 2 * 255, cv2.COLOR_RGB2BGR)
    file_name = 'data/GPEN/predict/test_img_predict.png'
    cv2.imwrite(file_name, sample)
    print(f"result saved in : {file_name}")
    print(f"\tFID: {fid_score}\n\tPSNR:{psnr_score}")
