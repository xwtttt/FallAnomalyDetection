import matplotlib.pyplot as plt
import numpy as np

x=np.arange(4)
#数据集
y1=[0.842122607,0.945568593,0.923314741,0.756167945]
y2=[0.97675405,0.985580509,0.970926682,0.966396704]
#误差列表
std_err1=[0.036152821,0.03156979,0.070709722,0.094287272]
std_err2=[0.015435763,0.011210651,0.035470286,0.051205111]
tick_label=['ACC','AUC','SEN','SPE']

error_params1=dict(elinewidth=3,ecolor='crimson',capsize=4)#设置误差标记参数
error_params2=dict(elinewidth=3,ecolor='blueviolet',capsize=4)#设置误差标记参数
#设置柱状图宽度
bar_width=0.4
#绘制柱状图，设置误差标记以及柱状图标签
plt.bar(x,y1,bar_width,yerr=std_err1,error_kw=error_params1,label='tag A')
plt.bar(x+bar_width,y2,bar_width,yerr=std_err1,error_kw=error_params2,label='tag B')

plt.xticks(x+bar_width/2,tick_label)#设置x轴的标签
#设置网格
plt.grid(True,axis='y',ls=':',color='r',alpha=0.3)
#显示图例
plt.legend()
#显示图形
plt.show()