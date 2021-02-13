import os
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from collections import Counter

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

PartOfSpeech_folder_name = "part_of_speech" #対象品詞の単語のみの抽出ファイルの出力先

########################################################
#################### main routine ######################
########################################################

for gensaki in gensaki_list:

  output_df = pd.DataFrame()
  print(f"Enumerating all words in {gensaki}")
  for yymmdd in tqdm(date_list):
    yymm = f"{int(yymmdd.year)-2000:02d}{int(yymmdd.month):02d}"
    data = pd.read_csv(f"{PartOfSpeech_folder_name}\\{yymm}{gensaki}_PartOfSpeech.csv",
                       engine="python", header=0, index_col=0)
    data = pd.DataFrame(pd.DataFrame(data).sum()).T
    data.index = [yymmdd]
    #単語毎の和を計算して、格納用のDFに縦に積み上げ
    output_df = pd.concat([output_df, data], sort=False)

  #結果の格納用
  output_df.to_csv(f"Word_TimeSeries_{gensaki}.csv", encoding="cp932")
