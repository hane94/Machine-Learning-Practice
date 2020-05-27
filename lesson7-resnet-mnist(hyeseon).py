from fastai.vision import *

path = untar_data(URLs.MNIST) #use minst dataset
path.ls()

il = ImageList.from_folder(path, convert_mode='L') #Imagelist - item list of images
il.items[0]

defaults.cmap='binary' #binary color map / set the default colormap for fastai / RGB가 아닌 다른 색상 표를 설정

il[0].show() #show the image

sd = il.split_by_folder(train='training', valid='testing') #train folder / validation folder

(path/'training').ls()

ll = sd.label_from_folder()

x,y = ll.train[0]

x.show()
print(y,x.shape)

tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], []) #small images of digit -> add a bit of random padding function

ll = ll.transform(tfms) #변환 호출 - 유효성 검사를 위한 세트 변환

bs = 128 #batch size

# not using imagenet_stats because not using pretrained model
data = ll.databunch(bs=bs).normalize() #no use pre-trained model in this case

x,y = data.train_ds[0]

x.show()
print(y)

def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8,8)) #we can get different positions of images with random padding

xb,yb = data.one_batch()
xb.shape,yb.shape

data.show_batch(rows=3, figsize=(5,5))

def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1) #input is 28 by 28

model = nn.Sequential(
    conv(1, 8), # 14 (grid size)
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16), # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32), # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16), # 2
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10), # 1
    nn.BatchNorm2d(10),
    Flatten()     # remove (1,1) grid
) #grid사이즈가 계속 절반으로 줄어듬

learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

print(learn.summary())

xb = xb.cuda()

model(xb).shape

learn.lr_find(end_lr=100)

learn.recorder.plot()

learn.fit_one_cycle(3, max_lr=0.1)

def conv2(ni,nf): return conv_layer(ni,nf,stride=2) #create cons batch normal value

model = nn.Sequential(
    conv2(1, 8),   # 14
    conv2(8, 16),  # 7
    conv2(16, 32), # 4
    conv2(32, 16), # 2
    conv2(16, 10), # 1
    Flatten()      # remove (1,1) grid
)

learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

learn.fit_one_cycle(10, max_lr=0.1)


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf, nf)
        self.conv2 = conv_layer(nf, nf)

    def forward(self, x): return x + self.conv2(self.conv1(x))

    help(res_block)

    model = nn.Sequential(
        conv2(1, 8),
        res_block(8),
        conv2(8, 16),
        res_block(16),
        conv2(16, 32),
        res_block(32),
        conv2(32, 16),
        res_block(16),
        conv2(16, 10),
        Flatten()
    )

    def conv_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), res_block(nf))

    model = nn.Sequential(
        conv_and_res(1, 8),
        conv_and_res(8, 16),
        conv_and_res(16, 32),
        conv_and_res(32, 16),
        conv2(16, 10),
        Flatten()
    )

    learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

    learn.lr_find(end_lr=100)
    learn.recorder.plot()

    learn.fit_one_cycle(12, max_lr=0.05)
    print(learn.summary())