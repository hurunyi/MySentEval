# MySentEval
- 下载infersent1.pkl模型和glove.840B.300d.txt到自定义的目录下。
- 运行 data/downstream/get_transfer_data.bash 来获取除了AFS（需要自己下载到data目录下）的所有数据集。
- 运行 pip install -r requirements.txt(python=3.6.12)配置所需环境。
- 切换到src目录下然后运行infersent.py即可（需要修改其中的路径为自定义的路径）。 