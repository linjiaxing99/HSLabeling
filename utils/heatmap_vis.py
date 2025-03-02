import sys

import numpy as np

sys.path.append('/home/isalab206/LJX/HSLabeling-master')
from utils import *
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from einops import rearrange, repeat
print("PyTorch Version: ", torch.__version__)
print('cuda', torch.version.cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0")
print('Device:', device)

def main():
    opt = Point_Options().parse()
    log_path, checkpoint_path, predict_path, _, _ = create_save_path(opt)
    model = Seg_Net(opt)
    if opt.resume:
        checkpoint = torch.load(checkpoint_path + '/model_best_2000.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print('resume success')
    model = model.to(device)
    model.eval()
    img_path = '/home/isalab206/Downloads/dataset/potsdam/train/vis_img/'
    label_path = '/home/isalab206/Downloads/dataset/potsdam/train/vis_surface/'
    save_path = '/home/isalab206/Downloads/dataset/potsdam/train/vis_save/'

    for filename in os.listdir(img_path):
        filename = '2_10_22.png'
        img = util.read(os.path.join(img_path, filename))[:, :, :3]
        img = util.Normalize(img, flag='potsdam')
        img = img[:, :, :opt.in_channels]
        img = img.astype(np.float32).transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        b, c, h, w = img.shape
        with torch.no_grad():
            input = img.cuda(non_blocking=True)
            output, feat = model(input)
            torch.cuda.synchronize()
        # feat = F.interpolate(feat, size=(int(h /4), int(w/4)), mode="bilinear", align_corners=True)
        _c = rearrange(feat, 'b c h w -> (b h w) c')
        # _c = feat_norm(feat_)
        _c = F.normalize(_c, p=2, dim=-1)
        # multil correlation, b=1
        corr = _c.matmul(_c.permute(1, 0))  # (b h w) c, c (b h w)
        # corr = corr.clamp(min=0)  # (b h w)^2
        corr = corr.view(feat.shape[2], feat.shape[3], feat.shape[2], feat.shape[3])


        # a = corr[120, 140, :, :]
        t_mask = np.zeros((feat.shape[2], feat.shape[3]), np.uint8)
        # cv2.line(t_mask, (0, 0), (30, 10), color=1, thickness=3)
        pts = np.array([[5,5],[5,12],[9,12],[9,5]], np.int32)
        pts = pts.reshape((-1, 2))
        cv2.fillPoly(t_mask, [pts], 1)
        tmp_indices = np.where(t_mask == 1)

        # label = util.read(os.path.join(label_path, filename))
        # if len(label.shape) == 3:
        #     label = label[:, :, 0]
        # label = cv2.resize(label.astype(np.float_), (768, 768))
        # label = cv2.resize(label.astype(np.float_), (feat.shape[2], feat.shape[3]), interpolation=cv2.INTER_AREA)
        # label[2, 2] =0

        # tmp_indices = np.where((label >= 0) & (label < 255))

        tmp_corr = corr[tmp_indices[0], tmp_indices[1]]  # n h w
        del corr
        heat, _ = torch.max(tmp_corr, dim=0)
        heat = torch.squeeze(heat)
        heat = heat.cpu().numpy()
        heat = cv2.resize(heat, (img.shape[2], img.shape[3]))
        # heat = np.ones((256, 256))
        heat = np.uint8(255 * heat)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        img = cv2.imread(img_path + filename)
        result_img = heat * 0.4 + img
        # result_img = heat
        cv2.imwrite(save_path + filename, result_img)


if __name__ == '__main__':
    main()

