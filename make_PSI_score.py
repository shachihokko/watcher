import os
import numpy as np
import pandas as pd

#ファイルがあるディレクトリで動かすためのおまじない
dir_org = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_org)

f_name = "result"

data = pd.read_csv(f"{f_name}.csv", engine="python",
                   header=0, index_col=0) #教師データの単語数え上げ行列の読み込み
data = data.T #row:category, column:word

Nc = data.sum(axis=1) # pd.Series: num of appearences in each category
W = len(data.columns) # num of unique words
Cat = len(data.columns) #Category

matrix_qwc = pd.DataFrame(index=list(data.index),
                          columns=list(data.columns))

#qwcの計算
for idx, line in enumerate(np.array(data)):
  tmp = np.array(line)
  matrix_qwc.iloc[idx, :] = (tmp+1)/(Nc[idx]+W)

#pcの計算
matrix_pc = (Nc+1)/(Nc.sum()+Cat)
matrix_pc = pd.DataFrame(matrix_pc, columns=["pc"])

#出力
matrix_qwc.to_csv("qwc_matrix.csv", encoding="cp932")
matrix_pc.to_csv("pc_matrix.csv", encoding="cp932")
