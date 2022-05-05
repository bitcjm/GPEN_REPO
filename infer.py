import argparse
import random

import cv2
import numpy as np
import paddle
import paddle.inference as paddle_infer
from data_loader.dataset_face import GFPGAN_degradation
from metrics import fid
from metrics.psnr import psnr


def preprocess(args):
    degrader = GFPGAN_degradation()

    img_gt = cv2.imread(args.img, cv2.IMREAD_COLOR)

    img_gt = cv2.resize(img_gt, (args.size, args.size), interpolation=cv2.INTER_NEAREST)

    img_gt = img_gt.astype(np.float32) / 255.
    img_gt, img_lq = degrader.degrade_process(img_gt)

    img_gt = (paddle.to_tensor(img_gt) - 0.5) / 0.5
    img_lq = (paddle.to_tensor(img_lq) - 0.5) / 0.5

    img_gt = img_gt.transpose([2, 0, 1]).flip(0).unsqueeze(0)
    img_lq = img_lq.transpose([2, 0, 1]).flip(0).unsqueeze(0)

    return np.array(img_lq).astype('float32'), np.array(img_gt).astype('float32')

def postprocess(input, target, output):
    input = paddle.to_tensor(input)
    target = paddle.to_tensor(target)
    output = paddle.to_tensor(output)

    FID_model = fid.FID(use_GPU=True)

    psnr_score = psnr(target, output)
    FID_model.update(output, target)
    fid_score = FID_model.accumulate()

    input = input.transpose([0, 2, 3, 1])
    target = target.transpose([0, 2, 3, 1])
    output = output.transpose([0, 2, 3, 1])
    sample_result = paddle.concat((input[0], output[0], target[0]), 1)
    sample = cv2.cvtColor((sample_result.numpy() + 1) / 2 * 255, cv2.COLOR_RGB2BGR)
    file_name = 'data/GPEN/predict/test_img_predict_infer.png'
    cv2.imwrite(file_name, sample)
    print(f"result saved in : {file_name}")
    print(f"\tFID: {fid_score}\n\tPSNR:{psnr_score}")


def create_predictor(model, params):
    config = paddle_infer.Config(model, params)
    config.disable_gpu()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = paddle_infer.create_predictor(config)

    return predictor

def parse_args():
    parser = argparse.ArgumentParser(description='Model infer')
    parser.add_argument("--model_file", type=str, help="model filename", default='outputs/GPEN/model.pdmodel')
    parser.add_argument("--params_file", type=str, help="parameter filename", default='outputs/GPEN/model.pdiparams')
    parser.add_argument('--img', dest='img', default='data/GPEN/predict/test_img.png', type=str)
    parser.add_argument('--size', dest='size', default=256, type=int)
    parser.add_argument('--is_concat', dest='is_concat', default=True, type=bool)
    return parser.parse_args()

def main():
    args = parse_args()

    paddle.seed(100)
    np.random.seed(100)
    random.seed(100)

    predictor = create_predictor(args.model_file, args.params_file)

    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor_list = []
    output_tensor_list = []
    for item in input_names:
        input_tensor_list.append(predictor.get_input_handle(item))
    for item in output_names:
        output_tensor_list.append(predictor.get_output_handle(item))
    input, target = preprocess(args)
    input_tensor_list[0].copy_from_cpu(input)
    predictor.run()
    output = output_tensor_list[0].copy_to_cpu()
    # Post process output
    postprocess(input, target, output)

if __name__ == "__main__":
    main()