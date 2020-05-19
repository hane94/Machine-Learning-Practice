#모듈 임포트

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn #기본 클래스 for neural network modules
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchsummary import summary


#이미지 트랜스폼
transform = transforms.Compose([
    transforms.ToTensor(), # ToTensor : numpy 배열의 이미지를 torch 텐서로 바꾸어준다
    transforms.Normalize((0.5, ), (0.5, )), #텐서 이미지 노말라이즈
])


batch_size = 128 #배치 사이즈 설정
z_dim = 100 #은닉 노드 수

database = dataset.MNIST('mnist', train = True, download = True, transform = transform)
#Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
train_loader = torch.utils.data.DataLoader(
    #dataset.MNIST('mnist', train = True, download = True, transform = transform),
    database,
    batch_size = batch_size, #128
    shuffle = True
)

for i, data in enumerate(train_loader): #train_loader index표시
    print ("batch id =" + str(i) )
    print (data[0])
    print (data[1])

def weights_init(m): #반환되는 클래스 이름에 따라 다른 초기화 값 설정
    classname = m.__class__.__name__ #해당 모듈의 이름을 출력
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) #Fills the input Tensor with values drawn from the normal distribution
        #torch.nn.init.normal_(Tensor,mean=0.0,std=1.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator_model(nn.Module): #분포 학습 모델
    def __init__(self, z_dim): #초기화 함수
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 7 * 7) #입력데이터에 대해서 선형 변환
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            #Applies a 2D transposed convolution operator over an input image composed of several input planes.
            #Transposed Convolution은 반대로 작동하는 Convolution. 보통 Convolution은 padding이 없을 시 feature를 뽑아내면서
            #이미지의 크기는 작아지게 되지만, Transposed Conv는 이미지의 크기를 더 크게 만든다
            #Transposed Convolution이 사용되는 모델의 종류 예시 : Autoencoder (인코더에서 이미지를 축소시키고 압축한 후 다시 복원할때)
            nn.BatchNorm2d(128),
            #Applies Batch Normalization over a 4D input
            nn.LeakyReLU(0.01), #(negative_slope = 0.01, inplace=False)
            #Applies the element-wise(요소별) function
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 256, 7, 7)
        return self.gen(x)

generator = Generator_model(z_dim).to(device)
generator.apply(weights_init)
summary(generator, (100, ))

class Discriminator_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01)
        )
        self.fc = nn.Linear(2048, 1)
    def forward(self, input):
        x = self.disc(input)
        return F.sigmoid(self.fc(x.view(-1, 2048)))

#GAN은 이미지 생성을 위한 2가지 모델을 동시에 사용 (Generator모델, discriminator모델)

discriminator = Discriminator_model().to(device)
discriminator.apply(weights_init) #초기 가중치 적용
summary(discriminator, (1, 28, 28))
criterion = nn.BCELoss()
#Creates a criterion that measures the Binary Cross Entropy between the target and the output
#딥러닝에서 분류 모델에 대한 손실 함수로 cross-entropy 사용
fixed_noise = torch.randn(64, z_dim, device=device)
#평균이 0이고 표준편차가 1인 가우시안 정규분포를 이용해 생성
#https://datascienceschool.net/view-notebook/4f3606fd839f4320a4120a56eec1e228/
doptimizer = optim.Adam(discriminator.parameters()) #PyTorch의 optim패키지
#모델의 가중치를 갱신할 optimizer정의
goptimizer = optim.Adam(generator.parameters())
real_label, fake_label = 1, 0

image_list = []
g_losses = []
d_losses = []
iterations = 0
num_epochs = 50

for epoch in range(num_epochs):
    print(f'Epoch : | {epoch + 1:03} / {num_epochs:03} |')
    for i, data in enumerate(train_loader):

        discriminator.zero_grad()

        real_images = data[0].to(device)  # real_images: size = (128,1,28,28)

        size = real_images.size(0)  # size = 128 = batch size
        label = torch.full((size,), real_label, device=device)  # real_label =1
        d_output = discriminator(real_images).view(-1) #Sets gradients of all model parameters to zero
        derror_real = criterion(d_output, label)

        derror_real.backward()

        noise = torch.randn(size, z_dim, device=device)  # noise shape = (128, 100)
        fake_images = generator(noise)  # fake_images: shape = (128,1,28,28)
        label.fill_(0)  # _: in-place-operation
        d_output = discriminator(fake_images.detach()).view(-1)

        derror_fake = criterion(d_output, label)
        derror_fake.backward()

        derror_total = derror_real + derror_fake
        doptimizer.step()

        generator.zero_grad()
        # label.fill_(real_images) #_: in-place-operation; the same as label.fill_(1)
        label.fill_(1)  # why is the label for the fake-image is one rather than zero?
        d_output = discriminator(fake_images).view(-1)
        gerror = criterion(d_output, label)
        gerror.backward()

        goptimizer.step()

        if i % 50 == 0:  # for every 50th i
            print(
                f'| {i:03} / {len(train_loader):03} | G Loss: {gerror.item():.3f} | D Loss: {derror_total.item():.3f} |')
            g_losses.append(gerror.item()) #g_losses의 값을 gerror.item()에 append
            d_losses.append(derror_total.item())

        if (iterations % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(train_loader) - 1)):
            with torch.no_grad():  # check if the generator has been improved from the same fixed_noise vector
                fake_images = generator(fixed_noise).detach().cpu()
            image_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

        iterations += 1 #반복

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses,label="Generator")
plt.plot(d_losses,label="Discriminator")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

for image in image_list:
    plt.imshow(np.transpose(image,(1,2,0)))
    plt.show()