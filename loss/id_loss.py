import os
import paddle
from paddle import nn
from loss.model_irse import Backbone
from paddle.vision.transforms import Resize
class IDLoss(paddle.nn.Layer):
    def __init__(self, base_dir='./', device='cuda', ckpt_dict=None):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')  ##需要改一下loss里的文件 
        if ckpt_dict is None:
            facenet_weights_path = os.path.join(base_dir, 'data/GPEN', 'model_ir_se50_2.pdparams')
            self.facenet.load_dict(paddle.load(facenet_weights_path))
        else:
            self.facenet.load_dict(ckpt_dict)
        self.face_pool = paddle.nn.AdaptiveAvgPool2D((112, 112))
        self.facenet.eval()
        

    def extract_feats(self, x):
        _, _, h, w = x.shape
        assert h==w
        ss = h//256
        x = x[:, :, 35*ss:-33*ss, 32*ss:-36*ss]  # Crop interesting region
        #x = self.face_pool(x)
        transform = Resize(size=(112, 112))
#         for key in self.facenet.parameters():
#             print(key.stop_gradient)
#             break
        for num in range(x.shape[0]):
            mid_feats = transform(x[num]).unsqueeze(0)
            if num == 0:
                x_feats = mid_feats
            else:
                x_feats = paddle.concat([x_feats, mid_feats], axis=0)

        x_feats = self.facenet(x_feats)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        loss_function = nn.L1Loss()
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])

            loss += 1 - diff_target

            count += 1

        return loss / count, sim_improvement / count, id_logs
