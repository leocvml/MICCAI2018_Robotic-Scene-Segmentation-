from mxnet.gluon import data as gdata
from mxnet import nd
from mxnet import gluon, image, ndarray
import mxnet as mx
########################################################################################################
###
###
### DATA LOADER
###
########################################################################################################
class SegDataset(gluon.data.Dataset):
    def __init__(self, root, resize, DataNameList, colormap=None, classes=None):
        self.root = root
        self.resize = resize
        self.colormap = colormap
        self.classes = classes
        self.DataNameList = DataNameList
        self.colormap2label = None
        self.load_images()

    def clsmap2channel(self,x):
        y = ndarray.one_hot(x, 11)
        y = ndarray.transpose(y,(2, 0, 1))
        return y
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
        return (data.astype('float32') / 127.5) - 1

    def __getitem__(self, item):
        if self.colormap is None:
            data = image.imresize(self.data[item], self.resize[0], self.resize[1])
            label = image.imresize(self.label[item], self.resize[0], self.resize[1])

            return data.transpose((2, 0, 1)), label.transpose((2, 0, 1))
        if self.colormap != None:
            data = image.imresize(self.data[item], self.resize[0], self.resize[1])
            label = image.imresize(self.label[item], self.resize[0], self.resize[1])
            sup_label = image.imresize(self.label[item], self.resize[0]//8, self.resize[1]//8)

            return data.transpose((2, 0, 1)), self.clsmap2channel(self.label_indices(label)),self.clsmap2channel(self.label_indices(sup_label))

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
        data_iter = gdata.DataLoader(dataset, batchsize, shuffle=True, last_batch='discard')

    return data_iter

dataset_dir ='../MICCAI_DATA/total640x512/'
batch_size = 3
resize = (640, 512)

with open('train_iter1.txt', 'r') as f:
    lines = f.readlines()

img_list = []
for line in lines:
    img_list.append(line[:-1])
training_list = img_list

train_iter = LoadDataset(dataset_dir, batch_size, resize, training_list, MICCAI_colormap,
                         MICCAI_classes)  # default is for 2 class if you want to multiclass
for d, l,sup_label in train_iter:
    break

print(d.shape)
print(l.shape)
print(sup_label.shape)

sup_label = ndarray.argmax(sup_label,axis=1)
def predict2img(predict):
    colormap = ndarray.array(MICCAI_colormap, ctx=mx.gpu(), dtype='uint8')  # voc_colormap

    target = predict.asnumpy()
    label = colormap[target[:, :, :]]
    return label.asnumpy()

from skimage import io

sup_label = predict2img(sup_label)
# print(sup_label.shape)
io.imshow(sup_label[0,:,:,:])
io.show()
