from mxnet import gluon, image, ndarray
from matplotlib import pyplot as plt
from mxnet.gluon import data as gdata
from os import listdir
from mxnet.gluon import nn
import mxnet as mx
from mxnet import nd
import os
import numpy as np


#####################################################################################
##
## DeepLab
##
####################################################################################
class stemblock(nn.HybridBlock):
    def __init__(self, filters):
        super(stemblock, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2D(self.filters, kernel_size=3, padding=1, strides=2)
        self.bn1 = nn.BatchNorm()
        self.act1 = nn.Activation('relu')

        self.conv2 = nn.Conv2D(self.filters, kernel_size=3, padding=1, strides=1)
        self.bn2 = nn.BatchNorm()
        self.act2 = nn.Activation('relu')

        self.conv3 = nn.Conv2D(self.filters, kernel_size=3, padding=1, strides=1)

        self.pool = nn.MaxPool2D(pool_size=(2, 2), strides=2)

    def hybrid_forward(self, F, x):
        stem1 = self.act1(self.bn1(self.conv1(x)))
        stem2 = self.act2(self.bn2(self.conv2(stem1)))
        stem3 = self.conv3(stem2)
        out = self.pool(stem3)

        return out

class conv_block(nn.HybridBlock):
    def __init__(self, filters):
        super(conv_block, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=3, padding=1),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=1),

            )

    def hybrid_forward(self, F, x):
        return self.net(x)

class DenseBlcok(nn.HybridBlock):
    def __init__(self, num_convs, num_channels):  # layers, growth rate
        super(DenseBlcok, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            for _ in range(num_convs):
                self.net.add(
                    conv_block(num_channels)
                )

    def hybrid_forward(self, F, x):
        for blk in self.net:
            Y = blk(x)
            x = F.concat(x, Y, dim=1)

        return x

class Deeplabv3(nn.HybridBlock):
    def __init__(self, growth_rate, numofcls):
        super(Deeplabv3, self).__init__()
        self.feature_extract = nn.HybridSequential()
        with self.feature_extract.name_scope():
            self.feature_extract.add(
                stemblock(256),
                DenseBlcok(8, growth_rate),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.conv1 = nn.HybridSequential()
        with self.conv1.name_scope():
            self.conv1.add(
                nn.Conv2D(128, kernel_size=1, strides=2),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self.conv3r6 = nn.HybridSequential()
        with self.conv3r6.name_scope():
            self.conv3r6.add(
                nn.Conv2D(128, kernel_size=3, strides=2, padding=6, dilation=6),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.conv3r12 = nn.HybridSequential()
        with self.conv3r12.name_scope():
            self.conv3r12.add(
                nn.Conv2D(128, kernel_size=3, strides=2, padding=12, dilation=12),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.conv3r18 = nn.HybridSequential()
        with self.conv3r18.name_scope():
            self.conv3r18.add(
                nn.Conv2D(128, kernel_size=3, strides=2, padding=18, dilation=18),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self.maxpool = nn.MaxPool2D(pool_size=2, strides=2)

        self.concatconv1 = nn.HybridSequential()
        with self.concatconv1.name_scope():
            self.concatconv1.add(

                nn.Conv2D(512, kernel_size=1),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

        self.feconv1 = nn.HybridSequential()
        with self.feconv1.name_scope():
            self.feconv1.add(
                nn.Conv2D(512, kernel_size=1),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.transUp = nn.HybridSequential()
        with self.transUp.name_scope():
            self.transUp.add(
                nn.Conv2DTranspose(256, kernel_size=4, padding=1, strides=2),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.decodeConv3 = nn.HybridSequential()
        with self.decodeConv3.name_scope():
            self.decodeConv3.add(
                nn.Conv2D(512, kernel_size=3, padding=1, strides=1),
                nn.BatchNorm(),
                nn.Activation('relu')
            )
        self.Up4 = nn.HybridSequential()
        with self.Up4.name_scope():
            self.Up4.add(
                nn.Conv2DTranspose(256, kernel_size=4, padding=1, strides=2),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2DTranspose(numofcls, kernel_size=4, padding=1, strides=2)
            )
        self.supervised = nn.HybridSequential()
        with self.supervised.name_scope():
            self.supervised.add(
                nn.Conv2D(numofcls, kernel_size=1, strides=1)
            )

    def hybrid_forward(self, F, x):
        out = self.feature_extract(x)
        conv1out = self.conv1(out)
        conv3r6out = self.conv3r6(out)
        conv3r12out = self.conv3r12(out)
        conv3r18out = self.conv3r18(out)
        maxpoolout = self.maxpool(out)

        second_out = ndarray.concat(conv1out, conv3r6out, conv3r12out, conv3r18out, maxpoolout, dim=1)

        encoder_out = self.concatconv1(second_out)
        sup_out = self.supervised(encoder_out)

        encoderUp = self.transUp(encoder_out)
        feconv1out = self.feconv1(out)
        combine_out = ndarray.concat(encoderUp, feconv1out, dim=1)
        output = self.decodeConv3(combine_out)
        output = self.Up4(output)

        weighting = F.UpSampling(sup_out, scale=8, sample_type='nearest')
        output = output * 0.5* weighting

        return output, sup_out

######################################################################################
###
### Discriminator
###
###
######################################################################################
class Discriminator(gluon.nn.HybridBlock):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Conv2D(64, kernel_size=3, strides=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2D(128, kernel_size=3, strides=2, padding=1),
                nn.InstanceNorm(),
                nn.LeakyReLU(0.2),
                nn.Conv2D(256, kernel_size=3, strides=2, padding=1),
                nn.InstanceNorm(),
                nn.LeakyReLU(0.2),
                nn.Conv2D(512, kernel_size=3, strides=2, padding=1),
                nn.InstanceNorm(),
                nn.LeakyReLU(0.2),
                nn.Conv2D(1, kernel_size=1, strides=1),
            )

    def hybrid_forward(self, F, x):
        return self.net(x)

####################################################################################
###
###
###  DataLoader
###
#####################################################################################
class SegDataset(gluon.data.Dataset):
    def __init__(self, root, resize, DataNameList, colormap=None, classes=None):
        self.root = root
        self.resize = resize
        self.colormap = colormap
        self.classes = classes
        self.DataNameList = DataNameList
        self.colormap2label = None
        self.load_images()

    def label_indices(self, img):
        if self.colormap2label is None:
            self.colormap2label = nd.zeros(256 ** 3)

            for i, cm in enumerate(self.colormap):
                self.colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        data = img.astype('int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return self.colormap2label[idx]

    def read_images(self, root):

        dataroot = root + 'left_frames/'  # left_frames   #data
        labelroot = root + 'labels/'  # labels   #label

        DataNamelist = sorted(self.DataNameList)

        data, label = [None] * len(self.DataNameList), [None] * len(self.DataNameList)

        for i, name in enumerate(DataNamelist):
            data[i] = image.imread(dataroot + name)
            label[i] = image.imread(labelroot + name)

        return data, label

    def load_images(self):
        data, label = self.read_images(root=self.root)
        self.data = [self.normalize_image(im) for im in data]
        if self.colormap is None:
            self.label = [self.normalize_image(im) for im in label]

        if self.colormap != None:
            self.label = label

        print('read ' + str(len(self.data)) + ' examples')

    def normalize_image(self, data):
        return data.astype('float32') / 255

    def __getitem__(self, item):
        if self.colormap is None:
            data = image.imresize(self.data[item], self.resize[0], self.resize[1])
            label = image.imresize(self.label[item], self.resize[0], self.resize[1])

            return data.transpose((2, 0, 1)), label.transpose((2, 0, 1))
        if self.colormap != None:
            data = image.imresize(self.data[item], self.resize[0], self.resize[1])
            label = image.imresize(self.label[item], self.resize[0], self.resize[1])
            sup_label = image.imresize(self.label[item], self.resize[0] // 8, self.resize[1] // 8)

            return data.transpose((2, 0, 1)), self.label_indices(label), self.label_indices(sup_label)

    def __len__(self):
        return len(self.data)

MICCAI_colormap = [[0, 0, 0], [0, 255, 0], [0, 255, 255], [125, 255, 12],
                   [255, 55, 0], [24, 55, 125], [187, 155, 25], [0, 255, 125],
                   [255, 255, 125], [123, 15, 175], [124, 155, 5]]
MICCAI_classes = ['background', 'shaft', 'clasper', 'wrist', 'kidney-parenchyma', 'covered-kidney',
                  'thread', 'clamps', ' suturing-needle', 'suction', 'small_intestine']
numcls = len(MICCAI_colormap)

def LoadDataset(dir, batchsize, output_shape, datalist, colormap=None, classes=None, train=True):
    dataset = SegDataset(dir, output_shape, datalist, colormap, classes)
    if not train:
        data_iter = gdata.DataLoader(dataset, 1, shuffle=False)
    if train:
        data_iter = gdata.DataLoader(dataset, batchsize, shuffle=True)

    return data_iter
########################################################################
###
###  predict to map
###
###
########################################################################
def predict2img(predict):
    colormap = ndarray.array(MICCAI_colormap, ctx=mx.gpu(), dtype='uint8')  # voc_colormap

    target = predict.asnumpy()
    label = colormap[target[:, :, :]]
    return label.asnumpy()

def PredictTrans(predict):
    colormap = ndarray.array(MICCAI_colormap, ctx=mx.gpu(), dtype='uint8')  # voc_colormap
    label = colormap[predict[:, :, :]]

    label = ndarray.transpose(label, (0, 3, 1, 2))

    label = label.astype(('float32'))
    return label

######################################################################
###
### learning rate decay
###
###
###
######################################################################
def Cosine_decay_schedule(current_step, total_step, warm_step, hold_base_step, learning_rate_base, warmup_rate_base):
    # learning_rate_base = 0.1
    # warmup_rate_base = 0.05
    # total_step = 1000
    # warm_step = 0
    # hold_base_step = 10

    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi * ((current_step - warm_step - hold_base_step) / (total_step - warm_step - hold_base_step))))

    if hold_base_step > 0:
        if current_step > warm_step + hold_base_step:
            learning_rate = learning_rate
        else:
            learning_rate = learning_rate_base

    if warm_step > 0:
        slope = (learning_rate_base - warmup_rate_base) / warm_step
        warmup_rate = slope * current_step + warmup_rate_base
        if current_step < warm_step:
            learning_rate = warmup_rate
        else:
            learning_rate = learning_rate

    return learning_rate


######################################################################################################################
import random

name = 'DLV3+3SGAN_selw'
dataset_dir = '../total640x512/'

result_folder = 'result_' + name + '/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
random.seed(0)

test_result_folder = 'test' + result_folder
if not os.path.exists(test_result_folder):
    os.makedirs(test_result_folder)

batch_size = 2
resize = (640, 512)

with open('train_iter1.txt', 'r') as f:
    lines = f.readlines()

img_list = []
for line in lines:
    img_list.append(line[:-1])
training_list = img_list

train_iter = LoadDataset(dataset_dir, batch_size, resize, training_list, MICCAI_colormap,
                         MICCAI_classes)  # default is for 2 class if you want to multiclass
for d, l, sup_label in train_iter:
    break

print(d.shape)
print(l.shape)
print(sup_label.shape)

# with open('test_iter1.txt','r') as f:
#     lines = f.readlines()
#
# img_list = []
# for line in lines:
#     img_list.append(line[:-1])
# testing_list = img_list
#
# test_iter = LoadDataset(dataset_dir, batch_size, resize,testing_list, MICCAI_colormap, MICCAI_classes,train=False)
# for d, l, sup_label in test_iter:
#     break
#
# print(d.shape)
# print(l.shape)
# print(sup_label.shape)



ctx = mx.gpu(0)
GenNet = nn.HybridSequential()
with GenNet.name_scope():
    GenNet.add(
        Deeplabv3(growth_rate=10, numofcls=numcls)  # 12
    )
GenNet.initialize()




DisNet = nn.HybridSequential()
with DisNet.name_scope():
    DisNet.add(
        Discriminator()
    )
DisNet.initialize()

softmax_CE = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
L2_loss = gluon.loss.L2Loss()





GenNet.collect_params().reset_ctx(ctx=ctx)
DisNet.collect_params().reset_ctx(ctx=ctx)

Gtrainer = gluon.Trainer(GenNet.collect_params(),'adam', {'learning_rate': 0.0002})
Dtrainer = gluon.Trainer(DisNet.collect_params(),'adam', {'learning_rate': 0.0002})


G_filename = name+'_G.params'
D_filename = name+'_D.params'


#GenNet.load_params(G_filename, ctx=ctx)


epochs = 300
warm_step = 3
hold_base = 180
base_lr = 0.0002
warm_lr = 0.000001

import time
from mxnet import autograd

for epoch in range(500):
    tic = time.time()
    for i, (d, l, sl) in enumerate(train_iter):
        x = d.as_in_context(ctx)
        y = l.as_in_context(ctx)
        sy = sl.as_in_context(ctx)

        with autograd.record():
            fake_y, sfake_y = GenNet(x)
            with autograd.pause():
                fake_y = ndarray.argmax(fake_y, axis=1)
                transfake_y = PredictTrans(fake_y)
                transfake_y = mx.ndarray.concat(x, transfake_y, dim=1)
                transreal_y = PredictTrans(y)
                transreal_y = mx.ndarray.concat(x, transreal_y, dim=1)

            real = DisNet(transreal_y)
            fake = DisNet(transfake_y)



            real_label = nd.ones_like(real,ctx=ctx)
            fake_label = nd.zeros_like(fake,ctx=ctx)


            errA_real = L2_loss(real, real_label)
            errA_fake = L2_loss(fake, fake_label)
            errD = (errA_real + errA_fake) * 0.5

        errD.backward()
        Dtrainer.step(x.shape[0])

        with autograd.record():
            fake_y,sfake_y = GenNet(x)
            pair_loss = softmax_CE(fake_y, y) + softmax_CE(sfake_y,sy)
            with autograd.pause():
                fake_y = ndarray.argmax(fake_y, axis=1)
                transfake_y = PredictTrans(fake_y)
                fake_y = mx.ndarray.concat(x,transfake_y, dim=1)



            real = DisNet(fake_y)

            errG = L2_loss(real, real_label) + 10 * pair_loss

        errG.backward()
        Gtrainer.step(x.shape[0])

        print(
            'Epoch %2d,batch %2d,G_loss %.5f ,D_loss %.5f , time %.1f sec' % (
                epoch,
                i,
                mx.ndarray.mean(errG).asscalar(),
                mx.ndarray.mean(errD).asscalar(),
                time.time() - tic))

    result, sresult = GenNet(x)
    result = ndarray.argmax(result, axis=1)
    result = predict2img(result)
    x = mx.ndarray.transpose(x, (0, 2, 3, 1))
    GT = predict2img(y)

   # figsize = (5, 3)
    # _, axes = plt.subplots(3, x.shape[0], figsize=figsize)
    # for n in range(x.shape[0]):
    #     axes[0][n].imshow(x[n].asnumpy())
    #     axes[1][n].imshow(result[n])
    #     axes[2][n].imshow(GT[n])
    #     axes[0][n].axis('off')
    #     axes[1][n].axis('off')
    #     axes[2][n].axis('off')
    axes[0][n].imshow(x[n].asnumpy())
    axes[1][n].imshow(result[n])
    axes[2][n].imshow(GT[n])
    axes[0][n].axis('off')
    axes[1][n].axis('off')
    axes[2][n].axis('off')


    plt.savefig(result_folder + str(epoch) + '.png')
    plt.close('all')


    GenNet.save_params(G_filename)
    DisNet.save_params(D_filename)


###########################################################################
###
### inference testiter
###
############################################################################

# from skimage import io
#
#
# def predict2img(predict):
#     colormap = ndarray.array(MICCAI_colormap, ctx=mx.gpu(), dtype='uint8')  # voc_colormap
#
#     target = predict.asnumpy()
#     label = colormap[target[:, :, :]]
#     return label.asnumpy()
#
# testing_list = sorted(testing_list)
#
#
# for i,(d, l, small_l) in enumerate(test_iter):
#     print(testing_list[i])
#     result,sup = GenNet(d.as_in_context(ctx))
#     result = ndarray.argmax(result, axis=1)
#     result = predict2img(result)
#     io.imsave(test_result_folder+testing_list[i],result[0, :, :, :])
#     #io.imshow(result[0, :, :, :])
#     #
#     # io.show()
#     #
#
#

# ################################################################################
# #####
# ####   Netweork inference
# ####
# ##############################################################################
# def predict2img(predict):
#     colormap = ndarray.array(test_iter._dataset.colormap, ctx=mx.gpu(), dtype='uint8')  # voc_colormap
#     # colormap = np.array(train_iter._dataset.colormap).astype('uint8')
#     target = predict.asnumpy()
#     label = colormap[target[:, :, :]]
#     return label.asnumpy()
#
# from skimage import io
#
# for d, l in test_iter:
#
#     d = d.as_in_context(ctx)
#     target = GenNet(d)
#     target = ndarray.argmax(target, axis=1)
#     predict = predict2img(target)
#
#     # d = ndarray.transpose(d, (0, 2, 3, 1))
#     # io.imshow(d[0, :, :, :].asnumpy())
#     # io.show()
#     # io.imshow(predict[0, :, :, :])
#     # io.show()
#     ######################################################################
#     ####    show image
#     ######################################################################
#     d = ndarray.transpose(d, (0, 2, 3, 1))
#
#     label = predict2img(l)
#
#     figsize = (10, 10)
#     _, axes = plt.subplots(3, d.shape[0], figsize=figsize)
#
#     for i in range(d.shape[0]):
#         axes[0][i].imshow(d[i, :, :, :].asnumpy())
#         axes[1][i].imshow(predict[i, :, :, :])
#         axes[2][i].imshow(label[i, :, :, :])
#
#     # axes(0,i).imshow(d[i, :, :, :].asnumpy())
#     # axes(1,i).imshow(predict[i, :, :, :])
#     # axes(2,i).imshow(label[i, :, :, :])
#
#     plt.show()


############################################################################
####
#### inference folder
####
#############################################################################
#
# from skimage import io
#
#
# def predict2img(predict):
#     colormap = ndarray.array(MICCAI_colormap, ctx=mx.gpu(), dtype='uint8')  # voc_colormap
#
#     target = predict.asnumpy()
#     label = colormap[target[:, :, :]]
#     return label.asnumpy()
#
# def process_image(fname):
#     with open(fname, 'rb') as f:
#         im = image.imdecode(f.read())
#     data = image.imresize(im, resize[0], resize[1])
#
#     data = (data.astype('float32')) / 255
#     return data.transpose((2,0,1)).expand_dims(axis=0)
#
#
# folder_path = 'D:/MICCAI_DATA/seq_1/left_frames/'
# save_path = 'seg_result/'
#
# datainfolder = [f for f in listdir(folder_path)]
#
# for dataname in datainfolder:
#     img = process_image(folder_path + dataname)
#     result = GenNet(img.as_in_context(ctx))
#     result = ndarray.argmax(result, axis=1)
#     result = predict2img(result)
#     io.imsave(save_path+dataname,result[0, :, :, :])
#     # io.imshow(result[0, :, :, :])
#     #
#     # io.show()
#     #