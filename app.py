from flask import *
import json
from utility import *
import tensorflow
from tensorflow.python.keras.models import load_model
from tensorflow import keras

app = Flask(__name__)

# 只接受POST方法访问
@app.route("/check_in", methods=["POST"])
def FaceMatch():
    # 默认返回内容
    return_dict = {'data': {
                            'name': 'None',
                            'identity' : None
                    },
                    'errty' : 'null',
                    'errmsg': "Null"}
    # 判断传入的json数据是否为空
    # if request.get_data() is None:
    #     return_dict['errty'] = 'The parameter is empty'
    #     return_dict['errmsg'] = 'Request to fill in the correct parameters '
    #     return json.dumps(return_dict, ensure_ascii=False)

    get_Data = request.get_data()

    get_Data = json.loads(get_Data)
    # print(get_Data)
    # print(type(get_Data))
    face = get_Data['face']
    mini_confidence = get_Data['min_confidence']
    print(mini_confidence)

    if float(mini_confidence) <= 0.6:
        return_dict['errty'] = 'Low confidence'
        return_dict['errmsg'] = 'This is picture with a low confidence to be a face'
        return json.dumps(return_dict, ensure_ascii=False)

    photo = FaceProcess(face)               #图片修改
    photo.FaceTrans()
    name, identity = photo.IdFace()         #返回id
    del photo
    data = {'name': name,
            'identity': identity
            }

    return json.dumps({"data": data}, ensure_ascii=False)



@app.route("/user/all/count", methods=["GET"])

def count_user():
    # 默认返回内容
    # return_dict = {'data': None,
    #                 'errty': 'null',
    #                 'errmsg': "Null"}

    user = UserImformation()
    data = user.count_user()
    print(data)
    del user
    return json.dumps({"data": data}, ensure_ascii=False)


@app.route("/user/all", methods=["GET"])

def get_all_user():

    # return_dict = {'data': None,
    #                'errty': 'null',
    #                'errmsg': "Null"}

    user = UserImformation()
    data = user.return_all_user()
    return json.dumps({"data": data})



@app.route("/user_register/upload", methods=["POST"])
def user_register_upload():
    # 默认返回内容
    get_Data = request.get_data()
    get_Data = json.loads(get_Data)
    faces = get_Data['faces']
    user = get_Data['user']
    print(faces)
    print(user)
    user_faces = FacesStorge(user['identity'], user['name'],faces)
    data = user_faces.add_user()
    if data == 1:
        return_dict = {'data': None,
                       'errty': 'id重复',
                        'errmsg': "id重复，写入数据失败"}
        return json.dumps({"data": return_dict})
    else:
        user_faces.write_images()

    del user_faces
    train_model()
    return json.dumps({"data": None})

@app.route("/user/delete", methods=["POST"])
def delete_user_byID():
    args = request.args.get('uid', '')
    delete_user(args)
    train_model()
    return json.dumps({"data":None})


@app.route("/admin/user/delete", methods=["POST"])
def context():
    # 默认返回内容
    return_dict = {'data': {
                            'name': 'None' ,
                            'identity' : None
                    },
                    'errty': 'null',
                    'errmsg': "Null"}
# @app.route("/admin/login", methods=["POST"])
# def context():
#     # 默认返回内容
#     return_dict = {'data': {
#                             'name': 'None' ,
#                             'identity' : None
#                     },
#                     'errty': 'null',
#                     'errmsg': "Null"}

if __name__ == "__main__":
    app.run(debug=True)

