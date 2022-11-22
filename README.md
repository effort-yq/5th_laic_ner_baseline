# 5th_laic_ner_baseline

第五届中国法研杯LAIC2022司法人工智能挑战赛，犯罪事实实体识别uie版本baseline，比赛官网：http://data.court.gov.cn/pages/laic.html

数据集，放在data文件夹中，uie代码参考大佬的github：https://github.com/heiheiyoyo/uie_pytorch

1、首先运行convert_data.py，将数据转成uie要求的格式，需要改convert_data.py中的数据路径；
2、从paddle中下载uie的模型，然后运行convert.py，具体可以参考上面参考大佬的GitHub解释；
3、运行finetune.py进行模型的训练，需要改模型路径和数据路径；
4、运行predict.py即可得到提交的文件。

记录自己学习的一个过程，有问题欢迎提出，谢谢！
