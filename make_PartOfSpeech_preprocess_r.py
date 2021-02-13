import os
import MeCab
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from collections import Counter
from tqdm import tqdm

#ファイルがあるディレクトリで動かすためのおまじない
dir_org = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_org)

########################################################
####################   setting   #######################
########################################################

sdate = dt.datetime.strptime("2020-11-01", "%Y-%m-%d") #対象データ始期
edate = dt.datetime.strptime("2021-01-01", "%Y-%m-%d") #対象データ終期
date_diff = relativedelta(edate, sdate)
months_diff = int(date_diff.years)*12 + int(date_diff.months)
date_list = [sdate+relativedelta(months=i) for i in range(months_diff+1)]

gensaki_list = ["gen", "saki"]

ParsedText_folder_name = "parse_text" #前処理済みのcsvファイルのフォルダ名
PartOfSpeech_folder_name = "part_of_speech" #対象品詞の単語のみの抽出ファイルの出力先


########################################################
####################   function   ######################
########################################################

def extract_target_PartOfSpeech(document: str) -> list:
  tokenizer = MeCab.Tagger("-Ochasen")
  tokenizer.parse("")
  node = tokenizer.parseToNode(document)
  keywords = []
  #品詞・活用分解した、文章を単語ごとにループし、欲しいものだけkeywordに追加
  while node:
    if node.feature.split(",")[0] == u"名詞":
      keywords.append(node.surface)
    elif node.feature.split(",")[0] == u"形容詞":
      keywords.append(node.feature.split(",")[6])
    elif node.feature.split(",")[0] == u"動詞":
      keywords.append(node.feature.split(",")[6])
    """
    elif node.feature.split(",")[0] == u"副詞":
      keywords.append(node.feature.split(",")[6])
    """
    node = node.next
  return keywords

########################################################
#################### main routine ######################
########################################################

### 全データから事前に活用系調整済みベースの単語情報を抽出 ###

for gensaki in gensaki_list:
  print(f"Extracting word in {gensaki}")

  for yymmdd in tqdm(date_list):
    yymm = f"{int(yymmdd.year)-2000:02d}{int(yymmdd.month):02d}"
    data = pd.read_csv(f"{ParsedText_folder_name}\\{yymm}{gensaki}.csv",
                       engine="python", header=0).loc[:,"comment"]

    #各コメントに対して、活用系調整済みベースの各単語を抽出
    output_df = pd.DataFrame() #結果の格納DF
    for idx, each_text in enumerate(data):
      #取得してきた文章出現順の単語リストを数え上げてDF化（列に単語、要素が出現数）
      tmp_df = pd.DataFrame(Counter(extract_target_PartOfSpeech(each_text)), index=[idx])
      #縦に積み上げ、このとき含まれていない単語はNaNとして処理される（出力時は空白になる）
      output_df = pd.concat([output_df, tmp_df], axis=0, sort=False)

    #結果の出力
    output_df.to_csv(f"{PartOfSpeech_folder_name}\\{yymm}{gensaki}_PartOfSpeech.csv", encoding="cp932")

