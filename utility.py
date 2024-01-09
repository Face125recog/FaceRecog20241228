import os
import re
import sys
import base64
from PIL import Image
import threading
from io import BytesIO
from databaseOp import DbOperate
import cv2
import numpy as np
import shutil
import tensorflow
from tensorflow.python.keras.models import load_model
from tensorflow import keras
from train_model import train_the_model

def RenameAllFirst(floder):
    fileList1 = os.listdir(floder)  #文件名字
    print("修改前：" + str(fileList1))
    currentpath = os.getcwd()
    os.chdir(floder)
    num = 1
    for fileName in fileList1:
        pat = ".+\.(jpg|png|gif|py|txt)"
        pattern = re.findall(pat, fileName)
        os.rename(fileName, "te"+(str(num)+'.'+pattern[0]))
        num = num + 1
    print("---------------------------------------------------------")
    os.chdir(currentpath)
    sys.stdin.flush()

def RenameAllLast(floder):
    fileList1 = os.listdir(floder)  # 文件名字
    print("修改前：" + str(fileList1))
    currentpath = os.getcwd()
    os.chdir(floder)
    num = 1
    for fileName in fileList1:
        pat = ".+\.(jpg|png|gif|py|txt)"
        pattern = re.findall(pat, fileName)
        os.rename(fileName, (str(num) + '.' + pattern[0]))
        num = num + 1
    print("---------------------------------------------------------")
    os.chdir(currentpath)
    sys.stdin.flush()

def LastFileNum(floderPath):
    if os.path.exists(floderPath):
        file_count = len(os.listdir(floderPath))
        return file_count
    else:
        return 0

#面部照片处理的类，用来将b64转换为图片，考虑是否把path改一下
class FaceProcess():
    def __init__(self, face):
        self.path = "temp.jpg"
        self.face = face


    #把base64编码转换我图片并保存到self.path路径上。
    def FaceTrans(self):
        head, context = self.face.split(",")
        image_data = base64.b64decode(context)
        img = Image.open(BytesIO(image_data))
        img.save(self.path)
        return image_data

    def IdFace(self):
        model = load_model('face_recognition.h5')  # 加载模型
        photo = cv2.imread(self.path)
        resized_photo = cv2.resize(photo, (100, 100))  # 调整图片大小
        recolord_photo = cv2.cvtColor(resized_photo, cv2.COLOR_BGR2GRAY)  # 调整为灰度图
        recolord_photo = recolord_photo.reshape((1, 1, 100, 100))
        result = model.predict(recolord_photo)  # 人物预测，返回试验集各个类别的概率
        max_index = np.argmax(result)  # print(max(result))
        db = DbOperate()
        result = db.find_who(max_index)
        db.close_connection()
        del db
        return result

    def path_tran(self, new_path):
        self.path = new_path


#存储功能的类
class FacesStorge():
    def __init__(self, id, name, faces:list):
        self.path = "faces/" + str(id-1)  #减去1，文件夹从0开始   数据库从1开始
        self.faces = faces
        self.id = id
        self.name = name
        self.db = DbOperate()

   # def faces_transform(self):
    def add_user(self):
        flag = self.db.add_employee(self.id, self.name)
        if flag:
            return 1


    def write_images(self):
        if os.path.exists(self.path):
            RenameAllFirst(self.path)  #route为文件夹路径
            RenameAllLast(self.path)
            start = LastFileNum(self.path)
        else:
            os.makedirs(self.path)
            start = 1

        for face in self.faces:
            face_trans = FaceProcess(face)
            new_path = self.path + '/' +str(start) + '.jpg'
            start = start + 1
            face_trans.path_tran(new_path)
            print(new_path)
            face_trans.FaceTrans()




class UserImformation():
    def __init__(self):
        self.employ = 'employee'

    def count_user(self):
        db = DbOperate()
        result = db.list_sum(self.employ)
        del db
        return result

#我懒
    def return_all_user(self):
        db = DbOperate()
        count = db.list_sum(self.employ)
        list = []
        index = 0
        while(index < count):
            name, identity = db.find_who(index)

            data = {'name': name,
                    'identity': identity
                    }

            list.append(data)
            index = index +1

        db.close_connection()
        del db
        return list

def delete_floder(path):  #删除该路径以及所有内部所有文件
    for filename in os.listdir(path):
        a_path = os.path.join(path, filename)
        if os.path.isfile(a_path) or os.path.islink(a_path):
            os.unlink(a_path)
        elif os.path.isdir(a_path):
            shutil.rmtree(a_path)
    os.rmdir(path)


def delete_user(id):
    db = DbOperate()
    flag = db.delete_employee(id)    #数据库删除
    path = "faces/"+str(int(id)-1)   #别忘了-1
    print(path)
    #flag为则 1 数据库无该数据 返回1 数据不存在
    if flag:
        return flag
    else:
        delete_floder(path)






def train_model():
    train_the_model("faces")

# globalState = {}
# mutex = threading.Lock()
#
# class State(object):
#     def __init__(self, name, id):
#         self.name = name
#         self.id = id
#         self.imgs = []
#
#     def addImg(self, img):
#         self.imgs.append(img)
#
#
# def getContext(name:str, id:int):
#     id = uuid4()
#     mutex.locked()
#     globalState[id] = State(name, id)
#     mutex.release()
#     return id
#
# def addImage(id:UUID, image):
#     mutex.locked()
#     state = globalState[id]
#     state.addImg(image)
#     mutex.release()
#
# def closeContext(id:UUID):
#     mutex.locked()
#     del globalState[id]
#     mutex.release()
