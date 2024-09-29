# coding:utf-8
__author__ = 'ila'

import re
import time,random
import pandas as pd
import numpy as np
import pymysql
from keras.layers import LSTM, Dense
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import History
class PredictUnknown(object):
    def __init__(self):
        host = '127.0.0.1'
        port = 3306
        user = 'root'
        passwd = '123456'
        db = 'springbootz3a8hl5t'
        charset = 'utf8mb4'

        self.con = pymysql.connect(
            host=host,
            port=port,
            user=user,
            passwd=passwd,
            database=db,
            charset=charset,
        )
        self.cur1 = self.con.cursor()
        self.cur2 = self.con.cursor()

        self.scaler=None
        self.dict_data={}

        self.raw_column_list = []
        self.column_list = []

    def process_data(self,sql):
        self.dict_data = {i: [] for i in self.raw_column_list}
        self.cur1.execute(sql)

        for data_set in self.cur1.fetchall():
            for idx, data in enumerate(data_set):
                self.dict_data[self.raw_column_list[idx]].append(data)

        self.cur1.close()

    # 构建时间序列预测模型
    def train(self,datas):

        X_train = np.array(datas).T.astype(float)  # 将数据转换为浮点数类型
        y_train = np.array(datas[-1]).astype(float)  # 将数据转换为浮点数类型

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = self.scaler.fit_transform(X_train)

        # 调整数据形状以适应 LSTM 模型
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

        model = Sequential()
        model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        history = History()
        model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2, callbacks=[history])
        model.save("xiaoliang.keras")
        return model

    # 预测
    def predict(self,model,data,):
        X_test = np.array(data).T.astype(float)
        X_test = X_test.reshape(1, -1)  # 将 X_test 转换为 2D 数组
        X_test = self.scaler.transform(X_test)
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        predicted_aqi = model.predict(X_test)
        return predicted_aqi

    def insert_data(self, data):
        select_sql=f'''select * from fengxianpinggu where yonghuzhanghao='{data.get("yonghuzhanghao")}' '''
        self.cur2.execute(select_sql)
        ret=self.cur2.fetchone()
        sql=""
        if ret is None or len(ret)<1:

            columns = []
            values = []
            for k, v in data.items():
                columns.append(f"{k}")
                values.append(f"{v}")
            sql = f'''insert into fengxianpinggu({','.join(columns)})values('{"','".join(values)}') '''
        else:
            list1=[]
            for k, v in data.items():
                list1.append(f"{k}='{v}' ")
            sql=f'''update fengxianpinggu set {",".join(list1)} where  yonghuzhanghao='{data.get("yonghuzhanghao")}' '''
        print(sql)
        self.cur2.execute(sql)
        self.con.commit()


if __name__ == "__main__":
    pu = PredictUnknown()

    pu.raw_column_list = ['yonghuzhanghao', "xingming", 'nianling', 'xingbie','shouji','touxiang','jibingxinxi','luruxinxi','lurushijian']
    pu.column_list = ['sex','age','gaoxueya','tangniaobing','xiyan']
    select_sql = f'select {",".join(pu.raw_column_list)} from jibingxinxi'

    pu.process_data(select_sql)

    train_data=[]
    sex_list=[]
    age_list=[]
    gaoxueya_list=[]
    tangniaobing_list=[]
    xiyan_list=[]
    for i in pu.dict_data.get("xingbie"):
        if '男' in i:
            sex_list.append(1)
        else:
            sex_list.append(2)

    for i in pu.dict_data.get("nianling"):
        c=re.compile("\d{1,3}")
        ret=c.findall(i)
        if len(ret)>0:
            age_list.append(ret[0])
        else:
            age_list.append(0)

    for i in pu.dict_data.get("jibingxinxi"):
        if '高血压' in i:
            gaoxueya_list.append(1)
        else:
            gaoxueya_list.append(0)
        if '糖尿病' in i:
            tangniaobing_list.append(1)
        else:
            tangniaobing_list.append(0)
        if '吸烟' in i:
            xiyan_list.append(1)
        else:
            xiyan_list.append(0)

    train_data.append(sex_list)
    train_data.append(age_list)
    train_data.append(gaoxueya_list)
    train_data.append(tangniaobing_list)
    train_data.append(xiyan_list)

    model = pu.train(train_data)

    for idx, col in enumerate(pu.dict_data.get("yonghuzhanghao")):
        data = []
        xingbie=1
        if '男' in pu.dict_data.get("xingbie")[idx]:
            data.append(1)#男1女2
        else:
            xingbie = 0
        data.append(xingbie)#男1女2

        nianling=0
        c = re.compile("\d{1,3}")
        ret = c.findall(pu.dict_data.get("nianling")[idx])
        if len(ret) > 0:
            nianling=ret[0]

        data.append(nianling)

        if '高血压' in pu.dict_data.get("jibingxinxi")[idx]:
            data.append(1) # 1有0无
        else:
            data.append(0)  # 1有0无
        if '糖尿病' in pu.dict_data.get("jibingxinxi")[idx]:
            data.append(1) # 1有0无
        else:
            data.append(0)  # 1有0无
        if '吸烟' in pu.dict_data.get("jibingxinxi")[idx]:
            data.append(1) # 1有0无
        else:
            data.append(0)  # 1有0无




        fengxianpinggu ="中"  if int(pu.predict(model, data)[0][0])>0 else "低"


        addtime= time.strftime('%Y-%m-%d', time.localtime(time.time()))
        yonghuzhanghao=pu.dict_data.get("yonghuzhanghao")[idx]
        xingming=pu.dict_data.get("xingming")[idx]

        shouji=pu.dict_data.get("shouji")[idx]
        touxiang=pu.dict_data.get("touxiang")[idx]
        jibingxinxi=pu.dict_data.get("jibingxinxi")[idx]
        luruxinxi=pu.dict_data.get("luruxinxi")[idx]
        lurushijian=pu.dict_data.get("lurushijian")[idx]
        pingguyijus=[]
        if '高血压' in pu.dict_data.get("jibingxinxi")[idx]:
            pingguyijus.append('高血压')
        if '糖尿病' in pu.dict_data.get("jibingxinxi")[idx]:
            pingguyijus.append('糖尿病')
        if '吸烟' in pu.dict_data.get("jibingxinxi")[idx]:
            pingguyijus.append('吸烟')

        insert_data = {
            "addtime": addtime,
            "yonghuzhanghao":yonghuzhanghao,
            'xingming':xingming,
            "nianling":nianling,
            "xingbie":"女" if xingbie in ["2",2] else "男",
            "shouji":shouji,
            "touxiang": touxiang,
            "jibingxinxi":jibingxinxi,
            "luruxinxi":luruxinxi,
            "lurushijian":lurushijian,
            "fengxianpinggu":f'{fengxianpinggu}',
            'pingguyiju':f"有以下习惯:{','.join(pingguyijus)}" if len(pingguyijus)>0 else "",
            'pinggushijian':addtime,
        }

        pu.insert_data(insert_data)

    pu.cur1.close()
    pu.cur2.close()
    pu.con.close()
    print("预测完毕,请刷新预测页面看结果")