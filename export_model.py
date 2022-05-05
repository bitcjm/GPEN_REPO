import os
import argparse
import paddle
from model.gpen import GPEN as MainModel

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument('--save_dir', dest='save_dir', help='The directory for saving the exported model', type=str,
                        default='outputs/GPEN')
    parser.add_argument('--model_path', dest='model_path', help='The path of model for export', type=str,
                        default='data/GPEN/G_256_weight_best.pdparams')
    #parser.add_argument('--model_path', dest='model_path', type=str, default='ckpts/G_256_repo_test.pdparams')
    parser.add_argument('--size', dest='size', default=256, type=int, help='the size of input and output images')
    parser.add_argument('--mul', dest='mul', default=1, type=int)
    parser.add_argument('--narrow', dest='narrow', default=0.5, type=int)
    parser.add_argument('--is_concat', dest='is_concat', default=True, type=bool)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = MainModel(args.size, 512, 8, channel_multiplier=args.mul, narrow=args.narrow, is_concat=args.is_concat)
    print(args.model_path)
    if os.path.exists(args.model_path):
        print('Loaded trained params of model successfully.')
        model.set_state_dict(paddle.load(args.model_path))
    else:
        print('Weight file dose not exist.')
    model.eval()

    input_spec = paddle.static.InputSpec(shape=[1, 3, args.size, args.size], dtype='float32', name='image')
    model = paddle.jit.to_static(model, input_spec=[input_spec])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(model, save_path)
    print(f'Model is saved in {args.save_dir}.')