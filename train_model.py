from os import listdir  # 如果需要修改分辨率，第13、113、114行代码处需要修改
import numpy as np
from PIL import Image
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils
import cv2

# 读取人脸图片数据，返回对应大小的以二维数组形式表示的照片img
def img2vector(fileNamestr):

    image= cv2.imread(fileNamestr)
    resized_photo = cv2.resize(image, (100, 100))  # 调整图片大小
    recolord_photo = cv2.cvtColor(resized_photo, cv2.COLOR_BGR2GRAY)  # 调整为灰度图
    recolord_photo = recolord_photo.reshape((1, 1, 100, 100))
    return recolord_photo


# 制作人脸数据集
def GetDataset(imgDataDir):
    print('| Step1 |: Get dataset...')
    FileDir = listdir(imgDataDir)  # 获取imgDataDir文件夹下的所有文件名以列表形式储存

    m = len(FileDir)  # 文件个数
    imgarray = []  # 一个空的二维数组，借助用于二维数组照片的存放
    hwLabels = []  # 每张照片主人的序号存放于此
    hwdata = []  # 存储每张照片的二维数组的一个三维数组

    for i in range(m):  # 逐个读取文件图片
        className = i
        subdirName = 'faces/' + str(FileDir[i]) + '/'  # 某个人所有照片统一存放的的文件夹
        fileNames = listdir(subdirName)  # 某个人所有照片的文件名
        lenFiles = len(fileNames)  # 每个人所拥有的照片的数量

        for j in range(lenFiles):  # 遍历每张照片
            fileNamestr = subdirName + fileNames[j]  # 每张照片的完整路径
            hwLabels.append(className)  # 照片主人的序号
            imgarray = img2vector(fileNamestr)  # 每张照片转化成二维数组
            hwdata.append(imgarray)  # 添加新的照片的二维数组

    hwdata = np.array(hwdata)
    return hwdata, hwLabels, 6  # 分别返回了所有人脸的二维数组、每个照片主人的序号、6是类别数


class MyCNN(object):
    FILE_PATH = "face_recognition.h5"  # 模型文件目录
    picHeight, picWidth = 57, 47  # 照片高57，宽47

    def __init__(self):
        self.model = None  # 创建空模型

    def read_trainData(self, dataset):  # 获取训练数据集
        self.dataset = dataset

    def build_model(self):  # 建立Sequential模型，并赋予参数
        print('| Step2 |: Init CNN model...')
        self.model = Sequential()  # 建立模型
        print('self.dataset.X_train.shape[1:]', self.dataset.X_train.shape[1:])  # 这里看不懂...
        self.model.add(Convolution2D(filters=32,  # 过滤器个数
                                     kernel_size=(5, 5),  # 卷积核尺寸
                                     padding='same',  # 边缘填充0
                                     # dim_ordering = 'th', 这个参数被注释掉了
                                     input_shape=self.dataset.X_train.shape[1:]  # 输入形状
                                     ))

        self.model.add(Activation('relu'))  # 激活函数为ReLU
        self.model.add(MaxPooling2D(pool_size=(2, 2),  # 池化核尺寸
                                    strides=(2, 2),  # 池化窗口移动步长
                                    padding='same'))  # 边缘0填充，若为valid则不填充
        self.model.add(Convolution2D(filters=64,
                                     kernel_size=(5, 5),
                                     padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))
        self.model.add(Flatten())  # 对多维数据进行降维
        self.model.add(Dense(512))  # 神经元节点数，输出的空间维度
        self.model.add(Activation('relu'))

        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()  # 定义完模型后，用该函数输出模型结构信息

    def train_model(self):  # 模型训练
        print('| Step3 |: Train CNN model...')
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(self.dataset.X_train,
                       self.dataset.Y_train,
                       epochs=10,  # 训练代次
                       batch_size=20)  # 每次训练样本数

    def evaluate_model(self):  # 显示loss和精度
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)
        print('| Step4 |: Evaluate performance...')
        print('============================================')
        print('Loss Value is:', loss)
        print('Accuracy Value is:', accuracy)

    def save(self, file_path=FILE_PATH):  # 保存模型
        print('| Step5 |: Save model...')
        self.model.save(file_path)
        print('Model ', file_path, 'is successfully saved.')


class DataSet(object):
    def __init__(self, path):
        self.num_classes = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.picWidth = 100
        self.picHeight = 100
        self.makeDataSet(path)

    def makeDataSet(self, path):
        imgs, labels, classNum = GetDataset(path)  # 根据指定路径读取出图片，标签，和类别数

        X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=1)  # 打乱分组

        X_train = X_train.reshape(X_train.shape[0], 1, self.picHeight, self.picWidth)/255.0
        X_test = X_test.reshape(X_test.shape[0], 1, self.picHeight, self.picWidth)/255.0

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        Y_train = np_utils.to_categorical(y_train, num_classes=classNum)
        Y_test = np_utils.to_categorical(y_test, num_classes=classNum)

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = classNum

def train_the_model(path):
    dataset = DataSet(path)
    model = MyCNN()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()
