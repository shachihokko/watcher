import os
import MeCab
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

########################################################
####################   setting   #######################
########################################################

#ファイルがあるディレクトリで動かすためのおまじない
dir_org = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_org)

sdate = dt.datetime.strptime("2000-01-01", "%Y-%m-%d") #対象データ始期
edate = dt.datetime.strptime("2021-01-01", "%Y-%m-%d") #対象データ終期
date_diff = relativedelta(edate, sdate)
months_diff = date_diff.years*12 + date_diff.months
date_list = [sdate+relativedelta(months=i) for i in range(months_diff+1)]

gensaki_list = ["gen", "saki"]

#前処理済みのcsvファイルのフォルダ名
#実行ファイルの１個下の階層に置いておくこと
parsedtext_folder_name = "parse_text"

#コメントから抽出した単語データの格納用のフォルダ名
#実行ファイルの１個下の階層に置いておくこと
#フォルダがなかったら作るみたいな処理は組み込んでないので
#必要フォルダは自分で作る
output_folder_name = "wordList_in_commnet"

########################################################
####################   function   ######################
########################################################

def extract_targets_PartsOfSpeech(document: str) -> list:
  tokenizer = MeCab.Tagger("-Ochasen")
  tokenizer.parse("")
  node = tokenizer.parseToNode(document)
  keywords = []
  #品詞・活用分解した、文章を単語ごとにループし、欲しものだけkeywordに追加
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
  print(f"Extracting words in {gensaki}")

  for yymmdd in tqdm(date_list):
    yymm = f"{yymmdd.years-2000:02d}{yymmdd.months:02d}"
    data = pd.DataFrame(f"{parsedtext_folder_name}\\{yymm}{gensaki}.csv",
                        engine="python", header=0).loc["comment"]

    #コメントの各文に対して、活用系調整済みベースの各単語を抽出
    output_list = []
    for each_text in data:
      #取得してきた単語をDFに格納するために","で結合して文字列にしてる
      join_words_str = ",".join(extract_targets_PartsOfSpeech(each_text))
      output_list.append(join_words_str)

    #出力
    output_df = pd.DataFrame(output_list, columns=["target_words"])
    output_df.to_csv(f"{output_folder_name}\\{yymm}{gensaki}_wordsList.csv",
                     encoding="cp932", index=False)

### 抽出してきた全単語を対象に時系列データを作成 ###
for gensaki in gensaki_list:

  #一旦全単語を取得しておく
  unique_words_list = []
  print(f"Enumerating all words in {gensaki}")
  for yymmdd in tqdm(date_list):
    yymm = f"{yymmdd.years-2000:02d}{yymmdd.months:02d}"
    data = pd.DataFrame(f"{output_folder_name}\\{yymm}{gensaki}.csv",
                        engine="python", header=0)
    for row in data:
      unique_words_list += row.split(",") #文字列にしてたのをバラす
    unique_words_list = list(set(unique_words_list)) #1回set型にして重複を削除しとく

  #結果の格納用
  output_df = pd.DataFrame(index=date_list, columns=unique_words_list)

  #各ファイルに対して全単語の数え上げ
  print(f"Making Timeseries of {gensaki}")
  for yymmdd in tqdm(date_list):
    yymm = f"{yymmdd.years-2000:02d}{yymmdd.months:02d}"
    data = pd.DataFrame(f"{output_folder_name}\\{yymm}{gensaki}.csv",
                        engine="python", header=0)

    used_words_list = []
    for row in data:
      used_words_list += row.split(",") #文字列にしてたのをバラす
    #各月のコメントにどの単語が何個あったかのリスト
    enumerate_list = [ used_words_list.count(word) for word in unique_words_list ]
    output_df.loc[[yymmdd], :] = enumerate_list #DFの同月に格納

  #結果の出力
  output_df.to_csv(f"Words_Timeseries_{gensaki}.csv", encoding="cp932")

