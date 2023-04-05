# 乳腺癌症分子分型预测网站

该软件是一个在线的乳腺癌症分子分型预测网站，用户可以上传自己的组学数据进行预测并得到结果。

## 安装

使用以下命令安装：

```shell
pip install breast-subtype-analysis
```

## 使用方法

- 命令行输入 `breast-subtype` 即可自动打开网站:
- 还要输入`model-log` 打开可视化界面:

```shell
breast-subtype
model-log
```


## 注意事项

- 确保输入的基因表达数据格式正确，详情请参考网站的示例文件。
- 确保端口号没有被占用，可以通过 `-p` 参数指定端口号。

## 开发者

- @夏淑凡

## 致谢

感谢以下开源项目：

- Flask
- scikit-learn
- pandas
- numpy
- torch
- snfpy
- model-log

## 许可证

MIT许可证

