import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import torch

def Gaussian_Distribution(N=2, M=1000, mean=0, cov=0):
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)

    return Gaussian


def generate_map(bbox, image_size, device='cuda'):
#     if isinstance(bbox, torch.Tensor):
#         pass
# #         if bbox.device=='cuda':
# #             bbox = bbox.reshape(-1, 4).detach().cpu().numpy()
# #         else:
# #             bbox = bbox.reshape(-1, 4).detach().cpu().numpy()
#     else:
    bbox = bbox.reshape(-1, 4)
    mask = np.zeros(image_size)
    
    for i in range(bbox.shape[0]):
        box_lt = [bbox[i][1], bbox[i][0]]
        h = bbox[i][2] - bbox[i][0]
        w = bbox[i][3] - bbox[i][1]
        
        # stepx = 1
        # stepy = w / h
        # center = [box_lt[0] + w / 2, box_lt[1] + h / 2]
        # X, Y = np.meshgrid(np.linspace(-stepx, stepx, image_size[0]), np.linspace(-stepy, stepy, image_size[1]))
        # d = np.dstack([X, Y])
        # mean = [center[0] / image_size[1] * 2 * stepx - stepx, center[1] / image_size[0] * 2 * stepy - stepy]
        # cov = np.eye(2)
        # cov[0][0] = 2 * stepx / (image_size[1] / w) / 6
        # cov[1][1] = 2 * stepy / (image_size[0] / h) / 6
        # Gaussian = Gaussian_Distribution(N=2, mean=mean, cov=cov)
        #
        # Z = Gaussian.pdf(d).reshape(*image_size)
        # mask[int(box_lt[1]):int(box_lt[1] + h + 20), int(box_lt[0]):int(box_lt[0] + w + 20)] = 1
        # Z = np.multiply(Z, mask)
        mask[int(box_lt[1]):int(box_lt[1] + h), int(box_lt[0]):int(box_lt[0] + w)] = 1
#         fig, ax = plt.subplots()
#         ax.imshow(mask, cmap='hot')
#         box = plt.Rectangle(box_lt, w, h, lw=5, alpha=0.5)

#         ax.add_patch(box)
#         plt.show()
    mask = torch.from_numpy(mask).float().to(device)
    return mask


if __name__ == "__main__":


    bbox = torch.Tensor([[[12.6, 547.8, 502.7, 899.1]]])
    image_size = [600, 900]
    generate_map(bbox, image_size)
