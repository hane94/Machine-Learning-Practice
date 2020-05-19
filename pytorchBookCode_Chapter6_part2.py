import torch

use_gpu = True if torch.cuda.is_available() else False #cuda 사용이 가능할 경우 GPU사용
#스타일간 모델
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
                       model_name='celebAHQ-512',
                       pretrained=True,
                       useGPU=use_gpu)


num_images = 5
noise, _ = model.buildNoiseData(num_images)

with torch.no_grad():
    generated_images = model.test(noise)


import matplotlib.pyplot as plt #모듈 임포트
import torchvision

grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
plt.imshow(grid.permute(1, 2, 0).cpu().numpy()) #그리드 안에 이미지