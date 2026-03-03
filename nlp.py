import jieba
from collections import Counter
import pandas as pd
from snownlp import SnowNLP
from snownlp import sentiment
from textblob import TextBlob

# 读取评论
with open('b站评论_20250201215118.csv', 'r', encoding='utf-8') as file:
    reports = file.readlines()

# 训练情感分析模型
sentiment.train('neg_all_dict.txt', 'pos_all_dict.txt') # 训练数据在github找的一个数据库，下载到本地目录
sentiment.save('aaasentiment.marshal') # 这是模型训练结果

# 用于存储分析结果的列表
positive_comments = []
negative_comments = []

for report in reports:
    report = report.strip() # 消除空格，这个是之前jieba库已经分好的结果，每条评论分词存为一个列表
    if report:
        # 使用SnowNLP进行极性分析，-1消极性，1积极性，0中性，数据是-1到1的小数
        t = SnowNLP(report)
        polarity_snownlp = t.sentiments
        # 使用TextBlob进行极性和主观性分析
        blob = TextBlob(report)
        polarity_textblob = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        subjectivity_label = '主观' if subjectivity > 0.5 else '客观'

        result = {
            '评论': report,
            'SnowNLP极性得分': polarity_snownlp,
            'TextBlob极性得分': polarity_textblob,
            '主观性得分': subjectivity,
            '主观性判断': subjectivity_label
        }

        if polarity_snownlp > 0.5:
            positive_comments.append(result)
        else:
            negative_comments.append(result)

# 创建DataFrame
positive_df = pd.DataFrame(positive_comments)
negative_df = pd.DataFrame(negative_comments)

# 写入Excel文件
with pd.ExcelWriter('ddd.xlsx') as writer:
    positive_df.to_excel(writer, sheet_name='正向情感评论', index=False)
    negative_df.to_excel(writer, sheet_name='负向情感评论', index=False)