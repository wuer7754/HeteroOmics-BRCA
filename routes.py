#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/11 0:48
# @Author : xia shufan
import base64
import glob
import logging
import pickle
import queue
import sqlite3
import webbrowser

import torch
from flask import Flask, send_from_directory, Response
from flask import request, send_file, session, flash, render_template, redirect, url_for
from torchviz import make_dot  # https://blog.csdn.net/qq_46343832/article/details/122494685
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.metrics import f1_score, recall_score, precision_score
from config import Config

app = Flask(__name__, template_folder='templates', static_folder='static')
# 添加配置信息
app.config.from_object(Config)  # 这个我还没测试出来，因为models.py 总是有问题
logger = logging.getLogger('my_logger')
logger.debug('This is a debug message.')


def get_db_connection():
    conn = sqlite3.connect('example.db')
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cur.fetchone()

        if user:
            flash('Email already exists', 'error')
        else:
            hashed_password = generate_password_hash(password, method='sha256')
            cur.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                        (name, email, hashed_password))
            conn.commit()
            flash('You are registered successfully', 'success')

        conn.close()
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cur.fetchone()

        if not user:
            flash('Invalid email or password', 'error')
        elif not check_password_hash(user['password'], password):
            flash('Invalid email or password', 'error')
        else:
            session['user_id'] = user['id']
            session['logged_in'] = True
            session['username'] = user['name']
            conn.close()
            return redirect(url_for('index'))

        conn.close()

    return render_template('login.html')


@app.route('/logout')
def logout():
    # 清除用户的登录状态并重定向到登录页面
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/')
def home():
    if session.get('logged_in'):  # 检查 session 中 logged_in 是否为 True
        return render_template('index.html')
    else:
        return redirect('/login')


@app.route('/index')
def index():
    if session.get('logged_in'):  # 检查 session 中 logged_in 是否为 True
        return render_template('index.html')
    else:
        return redirect('/login')


@app.route("/data_cleaning")
def render_page():
    if session.get('logged_in'):  # 检查 session 中 logged_in 是否为 True
        return render_template('data_cleaning.html')
    else:
        return redirect('/login')


@app.route("/download-zip")
def download_zip_page():
    if session.get('logged_in'):  # 检查 session 中 logged_in 是否为 True
        return render_template("download.html")
    else:
        return redirect('/login')


# 搞半天原来在服务器这边，download-zip/BreastData_1417.rar 都没实现


@app.route('/check_is_null', methods=['POST'])
def check_is_null():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    # 将上传的CSV文件转换为pandas DataFrame
    df = pd.read_csv(file, index_col=None, header=0)
    print(df.head)
    null_counts = df.isnull().sum()

    # 持久化null_counts数据，如果有空值
    if any(df):
        with open(os.path.join('static', 'tmp', 'nan_file.pkl'), 'wb') as f:
            pickle.dump(df, f)

    # 构造HTML表格来展示结果
    result_html = '<table>'
    for col, null_count in null_counts.items():
        result_html += f'<tr><td>{col}</td><td>{null_count}</td></tr>'
    result_html += '</table>'

    # 在HTML页面中呈现结果
    return render_template('null_clean_data.html', result=result_html)


from sklearn.impute import KNNImputer


@app.route('/clean_data', methods=['GET', 'POST'])
def clean_data():
    # Load null_counts data from pickle file
    with open(os.path.join('static', 'tmp', 'nan_file.pkl'), 'rb') as f:
        file = pickle.load(f)
    df = pd.DataFrame(file)
    print("进入clean_data()")
    print(df.head)
    print(df.isnull().sum())
    # Handle form submission
    if request.method == 'POST' and 'clean' in request.form:
        # Get checkbox values
        remove_null = bool(request.form.get('remove_null'))
        remove_duplicates = bool(request.form.get('remove_duplicates'))
        standardize = bool(request.form.get('standardize'))

        # Perform cleaning operations based on checkbox values
        # Here, we assume that `df` is your DataFrame that you want to clean
        if remove_null:
            count = 0  # 判断一共被删除了多少行
            for patient in df.columns:
                if df[patient].isnull().sum() / len(df[patient]) > 0.2:
                    df.drop(patient, axis=1)
                    count = count + 1
            for gene in df.index:
                if df.loc[gene].isnull().sum() / len(df.loc[gene]) > 0.2:
                    df.drop(gene, axis=0)
                    count = count + 1
            print("删除行数和列数加起来：{}".format(count))  # breast_methy : 397

        if remove_duplicates:
            df = df.drop_duplicates()
        if standardize:
            df = df.fillna(df.mean())
            # imputer = KNNImputer(n_neighbors=10)
            # df.loc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:])
            # Calculate new null counts
        # 将清除空值后的 DataFrame 保存为 nan_file.pkl 文件
        df.to_csv("static/clean_data/clean_data.csv")
        null_counts = df.isnull().sum()

        # Construct HTML table to display null counts
        result_html = '<table>'
        for col, null_count in null_counts.items():
            result_html += f'<tr><td>{col}</td><td>{null_count}</td></tr>'
        result_html += '</table>'

    # Render HTML template with result
    return render_template('null_clean_data.html', result=result_html)


@app.route('/process_null_data')
def process_data():
    if os.path.exists(f'static/clean_data/clean_data.csv'):
        path = f'static/clean_data/clean_data.csv'
        # file = open(path, 'rb') # 这个是用来显示的
        return send_file(path, mimetype='text/html', as_attachment=True)
    else:
        return "404"


import autoencoder_model
import torch
from flask import make_response
from matplotlib import pyplot as plt
import io
import torch.utils.data as Data


# Define the route for the autoencoder model
def work(Merge_data, in_feas, lr, bs, epochs, device, a, b, c, mode, topN, latent):
    print("现在进入work函数")
    # name of sample
    sample_name = Merge_data['Sample'].tolist()
    # change data to a Tensor
    X, Y = Merge_data.iloc[:, 1:].values, np.zeros(Merge_data.shape[0])
    TX, TY = torch.tensor(X, dtype=torch.float, device=device), torch.tensor(Y, dtype=torch.float, device=device)
    # train a AE model
    if mode == 0 or mode == 1:
        print('进入work函数的Training model...')
        Tensor_data = Data.TensorDataset(TX, TY)
        train_loader = Data.DataLoader(Tensor_data, batch_size=bs, shuffle=True)  # 设置每个批次bs大小，分批训练

        # initialize a model
        mmae = autoencoder_model.MMAE(in_feas, latent_dim=latent, a=a, b=b, c=c)
        mmae.to(device)
        mmae.train()  # 调用 train() 方法来将 mmae 对象设置为训练模式
        mmae.train_MMAE(train_loader, learning_rate=lr, device=device, epochs=epochs)
        mmae.eval()  # before save and test, fix the variables
        torch.save(mmae, 'static/model/AE/MMAE_model.pkl')  # 这种方式保存了mmae整个模型吗

        # load saved model, used for reducing dimensions
    if mode == 0 or mode == 2:
        print('进入到work函数的Get the latent layer output...')
        mmae = torch.load('static/model/AE/MMAE_model.pkl')  # 这里面到底是参数还是什么？
        omics_1 = TX[:, :in_feas[0]]
        omics_2 = TX[:, in_feas[0]:in_feas[0] + in_feas[1]]
        omics_3 = TX[:, in_feas[0] + in_feas[1]:in_feas[0] + in_feas[1] + in_feas[2]]
        latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3 = mmae.forward(omics_1, omics_2, omics_3)
        latent_df = pd.DataFrame(latent_data.detach().cpu().numpy())
        latent_df.insert(0, 'Sample', sample_name)
        # save the integrated data(dim=100)
        latent_df.to_csv('static/model/AE/latent_data.csv', header=True, index=False)
        # return render_template('process_data.html', processed_data=latent_df)  # render_template
    extract_features(Merge_data, in_feas, epochs, topN)
    return


from tqdm import tqdm


# @app.route("/extract_features", methods = ['POST', "GET"])
def extract_features(data, in_feas, epochs, topn):
    # extract features
    # get each omics data
    print('进入函数extract_features...')
    data_omics_1 = data.iloc[:, 1: 1 + in_feas[0]]
    data_omics_2 = data.iloc[:, 1 + in_feas[0]: 1 + in_feas[0] + in_feas[1]]
    data_omics_3 = data.iloc[:, 1 + in_feas[0] + in_feas[1]: 1 + in_feas[0] + in_feas[1] + in_feas[2]]

    # get all features of each omics data
    feas_omics_1 = data_omics_1.columns.tolist()
    feas_omics_2 = data_omics_2.columns.tolist()
    feas_omics_3 = data_omics_3.columns.tolist()

    # calculate the standard deviation of each feature
    std_omics_1 = data_omics_1.std(axis=0)
    std_omics_2 = data_omics_2.std(axis=0)
    std_omics_3 = data_omics_3.std(axis=0)

    # record top N features every 10 epochs
    topn_omics_1 = pd.DataFrame()
    topn_omics_2 = pd.DataFrame()
    topn_omics_3 = pd.DataFrame()

    # used for feature extraction, epoch_ls = [10,20,...], if epochs % 10 != 0, add the last epoch
    epoch_ls = list(range(10, epochs + 10, 10))
    if epochs % 10 != 0:
        epoch_ls.append(epochs)
    for epoch in tqdm(epoch_ls):
        # load model
        mmae = torch.load('static/model/AE/model_{}.pkl'.format(epoch))
        # get model variables
        model_dict = mmae.state_dict()  # 因为在实际应用中，我们可能需要将模型的参数进行保存和恢复，或者将模型参数传递给其他模型进行迁移学习等操作

        # get the absolute value of weights, the shape of matrix is (n_features, latent_layer_dim)
        weight_omics1 = np.abs(model_dict['encoder_omics_1.0.weight'].detach().cpu().numpy().T)
        weight_omics2 = np.abs(model_dict['encoder_omics_2.0.weight'].detach().cpu().numpy().T)
        weight_omics3 = np.abs(model_dict['encoder_omics_3.0.weight'].detach().cpu().numpy().T)

        weight_omics1_df = pd.DataFrame(weight_omics1, index=feas_omics_1)
        weight_omics2_df = pd.DataFrame(weight_omics2, index=feas_omics_2)
        weight_omics3_df = pd.DataFrame(weight_omics3, index=feas_omics_3)

        # calculate the weight sum of each feature --> sum of each row
        weight_omics1_df['Weight_sum'] = weight_omics1_df.apply(lambda x: x.sum(), axis=1)
        weight_omics2_df['Weight_sum'] = weight_omics2_df.apply(lambda x: x.sum(), axis=1)
        weight_omics3_df['Weight_sum'] = weight_omics3_df.apply(lambda x: x.sum(), axis=1)
        weight_omics1_df['Std'] = std_omics_1
        weight_omics2_df['Std'] = std_omics_2
        weight_omics3_df['Std'] = std_omics_3

        # importance = Weight * Std
        weight_omics1_df['Importance'] = weight_omics1_df['Weight_sum'] * weight_omics1_df['Std']
        weight_omics2_df['Importance'] = weight_omics2_df['Weight_sum'] * weight_omics2_df['Std']
        weight_omics3_df['Importance'] = weight_omics3_df['Weight_sum'] * weight_omics3_df['Std']

        # select top N features
        fea_omics_1_top = weight_omics1_df.nlargest(topn, 'Importance').index.tolist()
        fea_omics_2_top = weight_omics2_df.nlargest(topn, 'Importance').index.tolist()
        fea_omics_3_top = weight_omics3_df.nlargest(topn, 'Importance').index.tolist()

        # save top N features in a dataframe
        col_name = 'epoch_' + str(epoch)
        topn_omics_1[col_name] = fea_omics_1_top
        topn_omics_2[col_name] = fea_omics_2_top
        topn_omics_3[col_name] = fea_omics_3_top
    # all top N features
    topn_omics_1.to_csv('static/model/AE/topn_omics_1.csv', header=True, index=False)
    topn_omics_2.to_csv('static/model/AE/topn_omics_2.csv', header=True, index=False)
    topn_omics_3.to_csv('static/model/AE/topn_omics_3.csv', header=True, index=False)


@app.route("/ae_reduction")
def render_ae_page():
    if session.get('logged_in'):  # 检查 session 中 logged_in 是否为 True
        return render_template("AE.html")
    else:
        return redirect('/login')


@app.route('/run_ae_reduction', methods=['GET', 'POST'])
def train_ae():
    if request.method == 'POST':
        # Access the uploaded files and parameters
        file1 = request.files['omics-data-1']  # 获得file还不知道什么类型
        file2 = request.files['omics-data-2']
        file3 = request.files['omics-data-3']
        a = float(request.form['parameter-a'])
        b = float(request.form['parameter-b'])
        c = float(request.form['parameter-c'])
        lr = float(request.form['lr'])
        batch_size = int(request.form['batch_size'])
        epoch = int(request.form['epoch'])
        latent = int(request.form['latent'])
        topN = int(request.form['topn'])
        mode = int(request.form['mode'])
        # tmd ,总算写完了参数，真傻逼
        # 这一步不知道是否转化成功
        df_f1 = pd.DataFrame(file1)
        df_f2 = pd.DataFrame(file2)
        df_f3 = pd.DataFrame(file3)
        if not os.path.exists('df_f1.pkl') or not os.path.exists('df_f2.pkl') or not os.path.exists(
                'df_f3.pkl'):  # 如果有一个文件没有存在当前目录下，则说明文件都没存好
            # 将 DataFrame 对象序列化，并将其保存到本地文件系统或内存中
            with open('df_f1.pkl', 'wb') as f1, open('df_f2.pkl', 'wb') as f2, open('df_f3.pkl', 'wb') as f3:
                pickle.dump(df_f1, f1)
                pickle.dump(df_f2, f2)
                pickle.dump(df_f3, f3)
        else:
            with open('df_f1.pkl', 'rb') as f1, open('df_f2.pkl', 'rb') as f2, open('df_f3.pkl', 'rb') as f3:
                df_f1 = pickle.load(f1)
                df_f2 = pickle.load(f2)
                df_f3 = pickle.load(f3)
        # 选择共同的cols，其实就是同样数量的PATINENT_ID
        common_cols = df_f1.index.intersection(df_f2.index)
        common_cols = common_cols.intersection(df_f3.index)
        df_f1 = df_f1.loc[common_cols, :]
        df_f2 = df_f2.loc[common_cols, :]
        df_f3 = df_f3.loc[common_cols, :]  # 这里我想做个unittest

        # Rename columns Name to "Sample"
        df_f1.rename(columns={df_f1.columns.tolist()[0]: 'Sample'}, inplace=True)
        df_f2.rename(columns={df_f2.columns.tolist()[0]: 'Sample'}, inplace=True)
        df_f3.rename(columns={df_f3.columns.tolist()[0]: 'Sample'}, inplace=True)
        # sort values by sample
        df_f1.sort_values(by='Sample', ascending=True, inplace=True)
        df_f2.sort_values(by='Sample', ascending=True, inplace=True)
        df_f3.sort_values(by='Sample', ascending=True, inplace=True)
        # Check whether GPUs are available
        device = torch.device('cpu')
        # set random seed
        setup_seed(777)
        # dims of each omics data
        in_feas = [df_f1.shape[1] - 1, df_f2.shape[1] - 1, df_f3.shape[1] - 1]
        print("进入ae_reduction, train_ae()")
        print(in_feas)
        # merge the multi-omics data, calculate on common samples，这里也不知道合并的策略是什么，不相同的是不是舍弃了
        Merge_data = pd.merge(df_f1, df_f2, on='Sample', how='inner')
        Merge_data = pd.merge(Merge_data, df_f3, on='Sample', how='inner')
        Merge_data.sort_values(by='Sample', ascending=True,
                               inplace=True)  # 具体来说，在pd.merge()函数中使用inner参数，会返回具有匹配键值的两个数据集的交集。在结果中，只包含具有相同Sample值的记录。
        work(Merge_data, in_feas, lr=lr, bs=batch_size, epochs=epoch, device=device, a=a, b=b, c=c, mode=mode,
             topN=topN, latent=latent)
        return render_template('AE.html')
        # Redirect the user back to the upload file page
    return render_template(
        'AE.html')  # 就不用这个做法了return redirect('/upload_file')， 再写个路由，@app.route('/upload_file')def upload_file():return render_template('upload_file.html')


@app.route("/load_ae_model", methods=['POST', 'GET'])
def load_ae_model():
    with open('df_f1.pkl', 'rb') as f1, open('df_f2.pkl', 'rb') as f2, open('df_f3.pkl', 'rb') as f3:
        df_f1 = pickle.load(f1)
        df_f2 = pickle.load(f2)
        df_f3 = pickle.load(f3)
    with open("static/model/AE/MMAE_model.pkl", "rb") as file:
        mmae = torch.load(file)  # 这里面到底是参数还是什么？
        omics_1 = torch.tensor(df_f1.iloc[:, 1:].values, dtype=torch.float, device=torch.device('cpu'))
        omics_2 = torch.tensor(df_f2.iloc[:, 1:].values, dtype=torch.float, device=torch.device('cpu'))
        omics_3 = torch.tensor(df_f3.iloc[:, 1:].values, dtype=torch.float, device=torch.device('cpu'))
        latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3 = mmae.forward(omics_1, omics_2,
                                                                                      omics_3)  # 这个是我排查了很久的错误
        latent_df = pd.DataFrame(latent_data.detach().cpu().numpy())
        make_dot(mmae.forward(omics_1, omics_2, omics_3), params=dict(mmae.named_parameters())).render(
            "static/model/AE/ae_model")

    return render_template('process_data.html', processed_data=latent_df)


@app.route("/show_loss_figure", methods=['GET'])
def show_loss_figure():
    image_path = 'static/model/AE/AE_train_loss.png'

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            # 将二进制数据转换为base64编码格式的字符串
            encoded_data = base64.b64encode(image_data).decode('ascii')
        # 生成 HTML 响应
        html = f'''
               <html>
                   <head>
                       <title>Train Loss Figure</title>
                   </head>
                   <body>
                       <h1>Train Loss Figure</h1>
                        <img src="data:image/jpeg;base64,{encoded_data}"/>
                   </body>
               </html>
           '''

        # 返回响应
        return Response(html, mimetype='text/html')

    except IOError:
        return "图片路径不存在"


import os


@app.route("/show_topN_omics_x", methods=['GET', "POST"])
def show_topN_omics_x():
    csv = request.args.get('csv')
    print("进入到show_topN_omics_x()函数")
    print(csv)
    file_path = os.path.join('static', 'model', 'AE', csv)
    if os.path.exists(file_path):
        # with open(file_path, 'r') as f:
        #     omics_file = f.read()
        omics_file = pd.read_csv(file_path)
        return render_template('process_data.html', processed_data=omics_file)
    else:
        return "404"


@app.route('/show_ae_reduction_results', methods=['GET'])
def show_ae_reduction_results():
    if os.path.exists('static/model/AE/ae_model.pdf'):
        curr_dir = os.getcwd()
        file_path = "static/model/AE/ae_model.pdf"  # os.path.join(curr_dir,"ae_model.pdf") 这样的写法是不可以的，Not allowed to load local resource
        return render_template('pdf_viewer.html', curr_dir=curr_dir, filename=file_path)
    else:
        return 'PDF file not found!'


import pandas as pd


@app.route('/show_profile_report/<filename>')
def show_profile_report(filename):
    if os.path.exists(f'templates/profile_report/{filename}'):
        path = f"templates/profile_report/{filename}"
        file = open(path, 'rb')
        return send_file(file, mimetype='text/html')
    else:
        return "404"


@app.route("/show_short_csv/<filename>")
def show_short_csv(filename):
    if os.path.exists(f'static/tmp/{filename}'):
        path = f"static/tmp/{filename}"
        file = pd.read_csv(path)
        data = file.iloc[:200, :200]
        return render_template("process_data.html", processed_data=data)
    else:
        return "404"


# Define the route for downloading the latent data, 这个函数没写明白
@app.route('/download_latent_data/<filename>')
def download_latent_data(filename):  # make sure tag a's href = function name
    # latent_data = session.get('latent_data', None)
    # if latent_data is None:
    #     return redirect(url_for('upload_file'))
    #
    # # Convert the list to a CSV file and send it for download
    # output = io.StringIO()
    # writer = csv.writer(output)
    # writer.writerows(latent_data)
    # output.seek(0)
    return send_file(f'static/tmp/{filename}', mimetype='text/csv', as_attachment=True)


import snf  # pip install snfpy
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# 这个目前是按钮run snf 的功能
@app.route("/show-snf-csv", methods=['GET', 'POST'])
def show_snf_csv():
    fused_df = pd.read_csv("static/model/SNF/SNF_fused_matrix.csv", index_col=0)
    fused_df = fused_df.iloc[:200, :200]
    return render_template('process_data.html', processed_data=fused_df)


@app.route("/run-snf", methods=['GET', 'POST'])  # kao, methods
def run_snf():
    if request.method == 'POST':
        # Access the uploaded files and parameters
        file1 = request.files['data1']  # 获得file还不知道什么类型
        file2 = request.files['data2']
        file3 = request.files['data3']
        K_neighbors = int(request.form['k'])  # 在Flask中，从request.form中获取的数据都是字符串类型
        mu = float(request.form['mu'])
        distance = request.form['distance']
        df_f1 = pd.DataFrame(file1)
        df_f2 = pd.DataFrame(file2)
        df_f3 = pd.DataFrame(file3)
        if not os.path.exists('df_f1.pkl') or not os.path.exists('df_f2.pkl') or not os.path.exists(
                'df_f3.pkl'):  # 如果有一个文件没有存在当前目录下，则说明文件都没存好
            # 将 DataFrame 对象序列化，并将其保存到本地文件系统或内存中
            with open('df_f1.pkl', 'wb') as f1, open('df_f2.pkl', 'wb') as f2, open('df_f3.pkl', 'wb') as f3:
                pickle.dump(df_f1, f1)
                pickle.dump(df_f2, f2)
                pickle.dump(df_f3, f3)
        else:
            with open('df_f1.pkl', 'rb') as f1, open('df_f2.pkl', 'rb') as f2, open('df_f3.pkl', 'rb') as f3:
                df_f1 = pickle.load(f1)
                df_f2 = pickle.load(f2)
                df_f3 = pickle.load(f3)
                # 我想知道file1 对应的 是不是 cna
        df_f1.rename(columns={df_f1.columns.tolist()[0]: 'Sample'}, inplace=True)
        df_f2.rename(columns={df_f2.columns.tolist()[0]: 'Sample'}, inplace=True)
        df_f3.rename(columns={df_f3.columns.tolist()[0]: 'Sample'}, inplace=True)
        # sort values by sample
        df_f1.sort_values(by='Sample', ascending=True, inplace=True)
        df_f2.sort_values(by='Sample', ascending=True, inplace=True)
        df_f3.sort_values(by='Sample', ascending=True, inplace=True)
        print('Start similarity network fusion...')
        affinity_nets = snf.make_affinity(
            [df_f1.iloc[:, 1:].values.astype(np.float64),
             df_f2.iloc[:, 1:].values.astype(np.float64),
             df_f3.iloc[:, 1:].values.astype(np.float64)],
            metric=distance, mu=mu, K=K_neighbors
        )
        print('start fusing adjacency matrix...')
        fused_net = snf.snf(affinity_nets, K=K_neighbors)  # k is low digital ,neighbours
        print('start dataframe fused_net...')
        fused_df = pd.DataFrame(fused_net)
        fused_df.columns = df_f1['Sample'].tolist()
        fused_df.index = df_f1['Sample'].tolist()
        return render_template('process_data.html', processed_data=fused_df)
    return render_template('snf.html')


@app.route("/snf_network")
def show_page():
    if session.get('logged_in'):  # 检查 session 中 logged_in 是否为 True
        return render_template('snf.html')
    else:
        return redirect('/login')


@app.route("/read_snf_network", methods=['GET', 'POST'])
def snf_network_page():
    if request.method == 'POST':
        thread, results_queue = run_in_background()
        merge_result = results_queue.get()
        fused_df = pd.DataFrame(merge_result)
        return render_template('process_data.html', processed_data=fused_df)
    return render_template('snf.html')


def read_csv(filename, start, end):
    # Process the DataFrame here...
    return pd.read_csv(filename, nrows=end - start, skiprows=start)


def progress_large_csv(filename, request_queue):
    # large_file = 'static/model/SNF/SNF_fused_matrix.csv',这就是filename
    with ThreadPoolExecutor() as executor:
        futures = []  # none
        chunk_size = 1000
        large_file_size = os.path.getsize(filename)
        for i in range(0, large_file_size, chunk_size):
            start = i
            end = i + chunk_size
            future = executor.submit(read_csv, filename, start, end)
            futures.append(future)
        # wait for all thread to complete
        results_list = []
        for future in futures:  # executor.submit 是什么意思
            result = future.result()  # 需要加上括号 future 和 thread 到底是什么
            results_list.append(result)
        merge_result = pd.concat(results_list, ignore_index=True)
        request_queue.put(merge_result)  # 我现在都搞不清楚这行到底是放在with 里面还是 外面


import threading
from model_log.modellog import ModelLog
from gcn_model import GCN


def run_in_background():
    # 创建一个 request_queue,好像是用来进程通信
    request_queue = queue.Queue()
    # Start a new thread to process the CSV file
    thread = threading.Thread(target=progress_large_csv,
                              args=('static/model/SNF/SNF_fused_matrix.csv', request_queue))  # 者必须得用一个元组来显示args
    thread.start()
    return thread, request_queue  # 进程结束后不是自己返回thread, request_queue 之类的吗？


@app.route("/gcn_predict", methods=['GET', 'POST'])
def show_gcn_page():
    if session.get('logged_in'):  # 检查 session 中 logged_in 是否为 True
        return render_template('gcn_predict.html')
    else:
        return redirect('/login')


# 定义一个全局变量
GCN_model = None


@app.route("/run-gcn", methods=['GET', 'POST'])
def gcn_predict＿page():
    if request.method == "POST":
        autoencoder_file = request.files['autoencoder']
        snf_file = request.files['snf']
        labels = request.files['labels']
        test_sample_file = request.files['test_sample']
        threshold = float(request.form['threshold'])
        lr = float(request.form['lr'])
        epochs = int(request.form['epoch'])
        hidden = int(request.form['hidden'])
        nclass = int(request.form['nclass'])
        patience = int(request.form['patient'])
        dropout = float(request.form['dropout'])
        weight_decay = float(request.form['weight_decay'])
        device_type = request.form['device']
        device = torch.device('cuda' if device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
        # 设置随机种子，使每次调用随机数生成器时，都会生成相同的随机数序列，从而可以保证实验结果的可重复性。
        setup_seed(7)
        # autoencoder, snf, test file 都是 Werkzeug 中的 FileStorage 对象
        adj, data, labels = _load_data(snf_file, autoencoder_file, labels, threshold)
        print("第二步打印load_data输出的结果")
        print(adj)  # 这个值应该是多少
        print(data)
        print(labels)
        print("load_data结果打印结束！")
        # change dataframe to Tensor
        adj = torch.tensor(adj, dtype=torch.float, device=device)
        features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32, device=device)  # 这里挺复杂
        labels = torch.tensor(labels.iloc[:, 1].values, dtype=torch.long, device=device)
        print("打印change dataframe to tensor 的结果")
        print(adj)  # 这个值应该是多少
        print(features)
        print(labels)
        print("tensor 打印结果结束！")
        # 初始化 ModelLog
        # 初始化 ModelLog
        model_log = ModelLog('neu', 'Metabric多分类生存期预测')
        model_log.add_model_name('gcn神经网络addmethy')
        # batch_size = 900
        num_input = features.shape[1]
        params = {'learning_rate': lr,
                  'epochs': epochs,
                  'n_hidden_1': hidden,
                  'n_hidden_2': hidden,
                  'num_input': num_input,
                  'num_classes': nclass,
                  'dropout': dropout,
                  'weight decay': weight_decay,
                  'early_stop': patience}
        model_log.add_param(params, 'pytorch_params')
        logger.debug('Begin training model...')
        test_sample_file = pd.read_csv(test_sample_file, index_col=None, header=0)
        test_sample_df = pd.DataFrame(test_sample_file)
        # print(f"为什么没反应，test sample file shape : {test_sample_df.shape}")
        test_sample = test_sample_df.iloc[:, 0].tolist()
        print("检查test_sample_df 和 test_sample_list !")
        print(test_sample_df)
        print(test_sample)
        print("结束！")
        all_sample = data['Sample'].tolist()
        train_sample = list(set(all_sample) - set(test_sample))
        # get index of train samples and test samples
        train_idx = data[data['Sample'].isin(train_sample)].index.tolist()
        test_idx = data[data['Sample'].isin(test_sample)].index.tolist()
        # global GCN_model # 如果在函数内部重新定义一个已经存在的全局变量，Python会将这个变量视为局部变量，而不是全局变量.如果没有使用global关键字，Python
        # 会在函数内部创建一个新的局部变量，而不是修改全局变量的值。
        GCN_model = GCN(n_in=features.shape[1], n_hid=hidden, n_out=nclass, dropout=dropout)
        GCN_model.to(device)
        optim = torch.optim.Adam(GCN_model.parameters(), lr=lr, weight_decay=weight_decay)
        # Convert into tensor
        idx_train, idx_test = torch.tensor(train_idx, dtype=torch.long, device=device), \
                              torch.tensor(test_idx, dtype=torch.long, device=device)
        print("还没有进入test函数的idx_train,idx_test")
        print(idx_train)
        print(idx_test)
        print("结束！")
        '''
                save a best model (with the minimum loss value)
                if the loss didn't decrease in N epochs，stop the train process.
                N can be set by args.patience 
        '''
        loss_values = []  # record the loss value of each epoch
        bad_counter, best_epoch = 0, 0  # record the times with no loss decrease, and record the best epoch
        best = 1000  # record the lowest loss value
        for epoch in range(epochs):
            train_loss, train_acc = _train(device, GCN_model, epoch, optim, features, adj, labels, idx_train)
            loss_values.append(train_loss)
            test_loss, test_acc, f, recall, precision = _test(GCN_model, features, adj, labels, idx_test)
            # Model_log 添加评估指标
            model_log.add_metric('train_loss', train_loss, epoch)
            model_log.add_metric('test_loss', test_loss, epoch)
            model_log.add_metric('test_acc', test_acc, epoch)
            model_log.add_metric('test_F1', f, epoch)
            model_log.add_metric('test_recall', recall, epoch)
            model_log.add_metric('test_precision', precision, epoch)
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter = bad_counter + 1  # In this epoch, the loss value didn't decrease
            if bad_counter == patience:
                break
                # save model of this epoch
            torch.save(GCN_model.state_dict(), 'static/model/GCN/{}.pkl'.format(epoch))

            # reserve the best model, delete other models
            files = glob.glob('static/model/GCN/*.pkl')
            for file in files:
                name = file.split('\\')[1]
                epoch_nb = int(name.split('.')[0])
                # print(file, name, epoch_nb)
                if epoch_nb != best_epoch:
                    os.remove(file)
        # 结束训练，准备使用best model来预测，和画出模型图
        logger.debug("Training finished.")
        logger.debug(f"The best epoch model is {best_epoch}")
        """
                :param best_name:  str，最佳评估指标名称，
                :param best_value: float，最佳评估指标数值。
                :param best_epoch: int，训练周期

                添加当前模型训练中最佳的评估数据，一般放到模型训练的最后进行添加。
                """
        # model_log.add_best_result(best_name='best_loss', best_value=best, best_epoch=best_epoch)
        model_log.finish_model()  # 这是什么意思
        """
        关闭 SQLite 数据库连接
        """
        model_log.close()
        GCN_model.load_state_dict(torch.load("static/model/GCN/{}.pkl".format(best_epoch)))
        _predict(GCN_model, features, adj, all_sample, test_idx, labels)
        return '<iframe src="http://127.0.0.1:5432/" width="100%" height="500"></iframe>'
    return render_template("gcn_predict.html")


import torch.nn.functional as F


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _predict(GCN_model, features, adj, sample, idx, labels):
    """
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param sample: all sample names
    :param idx: the index of predict samples
    :param labels: the file named "breast__CLAUDIN_SUBTYPE_1471.csv" as an input, return from _load_data()
    :return:
    """
    GCN_model.eval()
    output = GCN_model.forward(features, adj)
    dot = make_dot(GCN_model.forward(features, adj), params=dict(GCN_model.named_parameters()))
    dot.render(filename="gcn-info", directory="static/model/GCN", format="pdf")
    predict_label = output.detach().cpu().numpy()
    predict_label = np.argmax(predict_label, axis=1).tolist()
    # predict_label = predict_label.iloc[idx, :] # 选择其中的test_sample idx, 这是list类型
    origin_label = labels[idx].tolist()
    # print("开始计算数组的长度")
    # print(len(predict_label))
    print(len(origin_label))
    res_data = pd.DataFrame(
        {"Sample": sample,
         "predict_label": predict_label,
         }
    )
    res_data = res_data.iloc[idx, :]  # 选择其中
    res_data["original_label"] = origin_label
    res_data.to_csv('static/model/GCN/GCN_predicted_data.csv', header=True, index=False)
    print("进入predict(), 已经执行完毕， 结束！")


def _train(device, model, epoch, optimizer, features, adj, labels, idx_train):
    '''
    :param epoch: training epochs
    :param optimizer: training optimizer, Adam optimizer
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param labels: sample labels
    :param idx_train: the index of trained samples
    '''
    print("进入train()函数 ，检查idx_train开始！")
    print(idx_train)
    print("结束检查")
    labels.to(device)
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # print(features)
    # print(output) # 到底是为什么都是nan，
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = _accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print('Epoch: %.2f | loss train: %.4f | acc train: %.4f' % (epoch + 1, loss_train.item(), acc_train.item()))
    return loss_train.data.item(), acc_train.item()  # 这里为什么要写item


def _accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def _test(model, features, adj, labels, idx_test):
    '''
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param labels: sample labels
    :param idx_test: the index of tested samples
    :return: loss_test, loss_train, f, recall, precision
    '''
    model.eval()
    output = model(features, adj)
    print("进入test() 函数， 开始打印output")
    print(output)
    print("开始打印idx_test")
    print(idx_test)
    print("结束打印！")
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])

    # calculate the accuracy
    acc_test = _accuracy(output[idx_test], labels[idx_test])

    # output is the one-hot label
    ot = output[idx_test].detach().cpu().numpy()
    # change one-hot label to digit label
    ot = np.argmax(ot, axis=1)
    # original label
    lb = labels[idx_test].detach().cpu().numpy()
    print('predict label: ', ot)

    print('original label: ', lb)

    # calculate the f1 score
    f = f1_score(ot, lb, average='weighted')
    recall = recall_score(ot, lb, labels=[0, 1, 2, 3, 4, 5, 6], average="macro")
    precision = precision_score(ot, lb, labels=[0, 1, 2, 3, 4, 5, 6], average='macro')
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    # return accuracy and f1 score
    return loss_test.data.item(), acc_test.item(), f, recall, precision


def _load_data(adj, fea, labels, threshold=0.005):
    #
    #         :param adj: the similarity matrix filename
    #         :param fea: the omics vector features filename
    #         :param lab: sample labels filename, 如果不训练，我们完全不需要处理label，但是我也不知道，下一步需不需要加上测试的功能
    #         :param threshold: the edge filter threshold
    #
    print("进入load data函数，打印开始")
    adj_df = pd.read_csv(adj, index_col=None, header=0)
    adj_df = pd.DataFrame(adj_df)
    fea_df = pd.read_csv(fea, index_col=None, header=0)  # 这里保留index ,同下一步idx_train保持一致
    fea_df = pd.DataFrame(fea_df)
    labels_df = pd.read_csv(labels, index_col=None, header=0)
    label_df = pd.DataFrame(labels_df)
    print("开始打印读取各种文件的结果")
    print(adj_df)
    print(fea_df)
    print(labels_df)
    print("结束！")
    if adj_df.shape[0] != fea_df.shape[0] or adj_df.shape[0] != label_df.shape[0]:
        app.logger('Input files must have same samples !')
        return "Input files must have same samples !"
    print("2.开始查看第一行第一个是否命名为Sample，以及有没有排好序")
    adj_df.rename(columns={adj_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    fea_df.rename(columns={fea_df.columns.tolist()[0]: 'Sample'}, inplace=True)
    label_df.rename(columns={label_df.columns.tolist()[0]: 'Sample'}, inplace=True)

    # align samples of different data
    adj_df.sort_values(by='Sample', ascending=True, inplace=True)
    fea_df.sort_values(by='Sample', ascending=True, inplace=True)
    label_df.sort_values(by='Sample', ascending=True, inplace=True)
    print(adj_df)
    print(fea_df)
    print(label_df)
    print("结束！")
    logger.debug('Calculating the  laplacian matix !')
    adj_m = adj_df.iloc[:, 1:].values  # 这个转化成了 array 格式的
    # print("adj_m的值是{}".format(adj_m))
    adj_m[adj_m < threshold] = 0

    # adjacency matrix
    exist = (adj_m != 0) * 1.0  # 乘1.0 实在做什么
    I = np.eye(adj_m.shape[0])
    exist = exist + I
    # calculate the degree matrix
    factor = np.ones(adj_m.shape[0])
    res = np.dot(exist, factor)
    diag_matrix = np.diag(res)
    d_inv = np.linalg.inv(diag_matrix)
    adj_hat = d_inv.dot(exist)
    return adj_hat, fea_df, label_df


@app.route("/show_snf_info", methods=["POST", "GET"])
def show_model_info():
    if os.path.exists('static/model/GCN/gcn-info.pdf'):
        file_path = "static/model/GCN/gcn-info.pdf"  # os.path.join(curr_dir,"ae_model.pdf") 这样的写法是不可以的，Not allowed to load local resource
        return render_template('pdf_viewer.html', filename=file_path)
    else:
        return 'PDF file not found!'


@app.route("/download-best-model", methods=["POST", "GET"])
def down_best_model():
    file_path = "static/model/GCN/194.pkl"  # 这file怎么是写死的
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "404"


@app.route("/download_predict_test_sample", methods=["POST", "GET"])
def download_predict_test_sample():
    # 读取第一个文件
    number_label_file = pd.read_csv(
        "static/model/GCN/GCN_predicted_data.csv")  # 我需要在这个数据表的“CLAUDIN_SUBTYPE” 后添加一列“RealType”
    # 获取不重复的CLAUDIN_SUBTYPE值和对应的PATIENT_ID
    sample_predict_label = number_label_file.groupby("original_label").apply(lambda x: x["Sample"].dropna().iloc[
        0])  # 这边不是去重key-values 中的values[['Sample','predict_label']].drop_duplicates().values
    dict_sample_predict_number = dict(zip(sample_predict_label.values, sample_predict_label.index))

    print(dict_sample_predict_number)
    # 读取第二个文件
    label_file = pd.read_csv(r"static\tmp\breast_patient.csv")
    patient_df = label_file[['PATIENT_ID', 'CLAUDIN_SUBTYPE']]

    patient_df["RealLabel"] = patient_df["PATIENT_ID"].apply(lambda x: dict_sample_predict_number.get(x))
    dict_subtype_number = dict(patient_df[['RealLabel', 'CLAUDIN_SUBTYPE']].dropna().values)
    print(dict_subtype_number)
    number_label_file["predict_real_label"] = number_label_file["predict_label"].apply(
        lambda x: dict_subtype_number.get(x))
    number_label_file['original_real_label'] = number_label_file['original_label'].apply(
        lambda x: dict_subtype_number.get(x))
    print(number_label_file)

    # 将predict_label和original_label进行比较，得到一个布尔值的DataFrame
    bool_df = number_label_file["predict_label"] != number_label_file["original_label"]

    def highlight_diff(row):
        if row['predict_label'] != row['original_label']:
            return ['background-color: rgb(173, 216, 230)'] * 5
        else:
            return ['background-color: rgb(70, 130, 180)'] * 5

    # 应用样式函数
    styled_df = number_label_file.style.apply(highlight_diff, axis=1)

    # 将结果转换为 HTML 页面
    html = styled_df.render()
    return html


@app.route("/show_KM_survival_analysis", methods=["POST", "GET"])
def show_km_analysis():
    file_path = "static/tmp/km_cox_5year_survival.zip"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "404"

import sys
import signal
import argparse



def my_exit(signum, frame):
    print()
    print('Good By!')
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, my_exit)
    signal.signal(signal.SIGTERM, my_exit)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=5431, help="指定端口号")
    args = parser.parse_args()

    try:
        webbrowser.open('http://127.0.0.1:%d/' % (args.p))
        app.run(debug=True, host="0.0.0.0", port=args.p)

    except Exception as e:
        print(str(args.p) + '端口已占用，请使用 model-log -p=5000 指定端口号，或关闭' + str(args.p) + '端口!')



if __name__ == '__main__':
    # conn = sqlite3.connect('example.db')
    # cur = conn.cursor()
    #
    # cur.execute('''
    #     CREATE TABLE users (
    #         id INTEGER PRIMARY KEY,
    #         name TEXT UNIQUE,
    #         email TEXT UNIQUE,
    #         password TEXT
    #     )
    # ''')
    # conn.commit()
    # conn.close()
    main() # 这里是程序的入口
