#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2/25/2023 2:44 AM
# @Author : xia shufan
class Config(object): # 这个我好像没用上~ 我就用了sqlite~
    #设置密匙要没有规律，别被人轻易猜到哦
    SECRET_KEY = 'a9087FFJFF9nnvc2@#$%FSD'
    # 上传文件别忘了设置哦
    UPLOAD_FOLDER = r'D:\Mogcn\BreastData_1417'
    # 配置数据库mysql的公式是, 等下，很复杂，开头居然是"mysql + pymysql :// username:pwd@localhost:3306/database?charset=utf8"
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:123456@localhost:3306/my_first_website?charset=utf8'
    # 如果你想用sqlite,就SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR,'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False