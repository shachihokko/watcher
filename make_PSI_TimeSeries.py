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
PSI_folder_name = "PSI_preprocess" #PSIに使用する単語のみの抽出ファイルの出力先
PSI_score_file_name = "PSI_score" #PSIに使用する単語と、それぞれのスコア

########################################################
####################   function   ######################
########################################################

def calc_PSI_category(word_appearance_mat: pd.DataFrame,
                      qwc_mat:pd.DataFrame, pc_mat:pd.DataFrame) -> (pd.Series, dict):
  #word_appearance - row: each comment, columns: word
  #qwc - row:category columns: word
  #pc - row:category columns: pc

  n_w = np.array(word_appearance_mat)

  each_cat_score_df = pd.DataFrame(columns=list(pc_mat.index))

  ### 対数で和に変換した上で、各行の文のスコアを計算
  for cat in list(pc_mat.index):

    qwc = np.array(qwc_mat.loc[[cat], :])
    pc = np.array(pc_mat.loc[[cat], :])

    """
    (1*#ofword)*(#ofcomment*#ofword)形式で計算ブロードキャストを利用して
    (1*#ofword)*(1*#ofword)の要素ごとの計算を#ofcomment回実施
    その後ブロードキャスト利用して、各行にterm2を足してる
    """
    each_cat_score_df[cat] = np.sum((n_w)*np.log(qwc), axis=1) + float(np.log(pc))

  each_comment_category = each_cat_score_df.idxmax(axis=1) #各行のargmaxを計算
  dict_category_num = Counter(list(each_comment_category)) #Counterを用いて、各カテゴリーの数を算出

  return dict_category_num, each_comment_category

########################################################
#################### main routine ######################
########################################################

###PSIのスコアリング読み込み
#row:category, columns:word
qwc_mat = pd.read_csv(f"{PSI_folder_name}\\qwc_matrix.csv",
                       engine="python", header=0, index_col=0)
#row:category, columns:1
pc_mat = pd.DataFrame(pd.read_csv(f"{PSI_folder_name}\\pc_matrix.csv",
                     engine="python", header=0, index_col=0))

PSI_word = list(qwc_mat.columns) #PSIに使用する単語のリスト
PSI_Cat = list(qwc_mat.index) #PSIのカテゴリー

for gensaki in gensaki_list:
  print(f"PSI_TimeSeries of {gensaki}")
  output_df = pd.DataFrame(index=date_list, columns=PSI_Cat) #PSIの結果格納用

  for yymmdd in tqdm(date_list):
    yymm = f"{int(yymmdd.year)-2000:02d}{int(yymmdd.month):02d}"
    data = pd.read_csv(f"{PartOfSpeech_folder_name}\\{yymm}{gensaki}_PartOfSpeech.csv",
                        engine="python", header=0, index_col=0)

    data_PSI = data.loc[:, PSI_word] #PSIに使用する単語部分だけを抽出
    data_PSI = data_PSI.fillna(0) #単語出現しなかった際のnaを0で埋める
    dict_category_num, each_comment_category = calc_PSI_category(data_PSI, qwc_mat, pc_mat)

    output_df.loc[[yymmdd], list(dict_category_num.keys())] = list(dict_category_num.values()) #結果の格納

    ###元データにPSI_Categoryをつけて出力
    org_data_col_names = list(data.columns)
    data["PSI_Category"] = each_comment_category
    new_columns = ["PSI_Category"] + org_data_col_names
    data = data[new_columns] #PSIのカテゴリーが先頭列になるように並べ替え
    data.to_csv(f"{PSI_folder_name}\\{yymm}{gensaki}_PSI_Category.csv",
                encoding="cp932")

  #PSIの時系列の出力
  output_df = output_df.loc[:, PSI_Cat]
  output_df.to_csv(f"PSI_TimeSeries_{gensaki}.csv", encoding="cp932")

