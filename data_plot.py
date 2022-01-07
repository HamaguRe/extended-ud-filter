# グラフ作成

import csv
import matplotlib.pyplot as plt

# Figureを追加
fig = plt.figure(figsize = (8, 8))

ax1 = fig.add_subplot(411, ylabel='x1')  # 3行1列1番目
ax2 = fig.add_subplot(412, ylabel='x2')
ax3 = fig.add_subplot(413, ylabel='x3')
ax4 = fig.add_subplot(414, xlabel='time', ylabel='y')

# 時刻
t = []
# 出力の真値
y_true = []
# 出力の観測値
y = []
# 状態変数の真値
x = [[], [], []]
# EKF推定値
xhat = [[], [], []]

# CSVからデータを読み出してnumpy配列に追加
with open('./result.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        nums = [float(v) for v in row]  # 文字列から浮動小数点数に変換

        # 時刻
        t.append(nums[0])
        # 出力の真値
        y_true.append(nums[1])
        # 出力の観測値
        y.append(nums[2])
        for i in range(3):
            x[i].append(nums[i+3])
            xhat[i].append(nums[i+6])

# 描画
ax1.plot(t, x[0],    label="Reference", color="black") # 真値
ax1.plot(t, xhat[0], label="Estimated", color="red", linestyle = "--") # EKF推定値
ax2.plot(t, x[1],    label="Reference", color="black")
ax2.plot(t, xhat[1], label="Estimated", color="red", linestyle = "--")
ax3.plot(t, x[2],    label="Reference", color="black")
ax3.plot(t, xhat[2], label="Estimated", color="red", linestyle = "--")
ax4.plot(t, y_true,  label="Reference", color="black")
ax4.plot(t, y, label="Observed", color="red", linestyle = "--")
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()