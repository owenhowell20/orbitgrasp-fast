import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter
import pandas as pd

sns.set(style="darkgrid")
# plt.style.use('ggplot')

data_w_aug_wo_crop = [
    [0.5185232208244931, 0.7303742155286407, 0.7489481065918654],
    [0.4967859243576133, 0.7522548427295885, 0.7639083683964469],
    [0.4857090721555601, 0.7514003069741305, 0.7788686302010285],
    [0.47374414076724725, 0.7638661970026084, 0.7858812529219261],
    [0.469562575940344, 0.766771361956311, 0.793361383824217],
    [0.4780495379336941, 0.7668216238586408, 0.7854137447405329],
    [0.47148306453195604, 0.7716055654508313, 0.7793361383824217],
    [0.47395835849341106, 0.765601806715305, 0.7798036465638148],
    [0.47046680889445497, 0.7730951878569181, 0.7886863020102852],
    [0.471534061604826, 0.7763479685409986, 0.7816736792893876],
    [0.47609157432367055, 0.7775242729700297, 0.7877512856474989],
    [0.4765589623435524, 0.7762401223865387, 0.7793361383824217],
    [0.49099179845635943, 0.7760161374037708, 0.7812061711079944],
    [0.4910116263592547, 0.7774812409244112, 0.7830762038335671],
    [0.4935213154760486, 0.7813365667459283, 0.7858812529219261]
]

# Convert to DataFrame
df_w_aug = pd.DataFrame(data_w_aug_wo_crop).round(3)
valid_loss_w_aug_wo_crop = df_w_aug[0].tolist()
valid_acc_w_aug_wo_crop = df_w_aug[1].tolist()
valid_max_w_aug_wo_crop = df_w_aug[2].tolist()

ours = [
    [0.334497665842877, 0.850764126353377, 0.8764249886000912],
    [0.3257062924924747, 0.8577925997730591, 0.8887368901048791],
    [0.3090899241294048, 0.8655764173480185, 0.898312813497492],
    [0.31881005532370277, 0.8609746658484032, 0.8887368901048791],
    [0.3017515923940002, 0.8683474694301807, 0.896032831737346],
    [0.3074349737027554, 0.8674337416683437, 0.9010487916096671],
    [0.30658379601713015, 0.8674480806351578, 0.903328773369813],
    [0.3030730723711287, 0.8718026256621074, 0.8969448244414044],
    [0.28185531756045645, 0.8806947671519095, 0.9019607843137255],
    [0.28434913308048393, 0.8796038007116513, 0.9037847697218422],
    [0.2946801747934259, 0.8794987074762406, 0.905608755129959],
    [0.29057439028664894, 0.8817645231991449, 0.9015047879616963],
    [0.3013067330059422, 0.878537388918501, 0.9042407660738714],
    [0.29573932062882863, 0.885343466910444, 0.9088007295941632],
    [0.2915501353449608, 0.885270716811879, 0.9088007295941632]
]

df_ours = pd.DataFrame(ours).round(3)
valid_loss_ours = df_ours[0].tolist()
valid_acc_ours = df_ours[1].tolist()
valid_max_ours = df_ours[2].tolist()

colors = ['#00ccff', '#ffcc00', '#66ff66', '#ff6666']
colors = ['orange', 'green', 'blue', 'red']

# sns.set(style='whitegrid')
# sns.set_style("darkgrid", {"axes.facecolor": ".96"})
# sns.set_context("talk", font_scale=1.1)
plt.figure(figsize=(10, 7))
ax = plt.gca()
ax.set_xticks(range(0, 15, 2))  # 原始刻度为 0, 2, 4, ..., 14
ax.set_xticklabels([str(i+1) for i in range(0, 15, 2)])
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(valid_loss_w_aug_wo_crop, label='no larger neighborhoods', color=colors[2], linewidth=2, marker='o',
         markersize=6)
# plt.plot(valid_loss_wo_aug_w_crop, label='no data aug', color=colors[1], linewidth=2, marker='x',markersize=8)
# plt.plot(valid_loss_wo_aug_wo_crop, label='neither', color=colors[2], linewidth=2, marker='s',markersize=6)
plt.plot(valid_loss_ours, label='ours', color=colors[3], linewidth=3, marker='d', markersize=6)

plt.legend(loc='best', fontsize=16)
plt.title('Validation Loss for Single-view Training', fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, linewidth=1.8)
sns.despine()
# ax.set_xticks(range(-1, 16, 2))
# ax.set_xlim(-1, 15)

plt.savefig('loss_comparison.png', format='png', dpi=500, bbox_inches='tight')
plt.show()

# sns.set(style='whitegrid')
# sns.set_style("darkgrid", {"axes.facecolor": ".96"})
# sns.set_context("talk", font_scale=1.1)
plt.figure(figsize=(10, 7))
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
formatter = FuncFormatter(lambda x, pos: f'{x:.2f}')
ax.yaxis.set_major_formatter(formatter)
plt.plot(valid_acc_w_aug_wo_crop, label='no larger neighborhoods', color=colors[2], linewidth=2, marker='o',
         markersize=6)
# plt.plot(valid_acc_wo_aug_w_crop, label='no data aug', color=colors[1], linewidth=2, marker='x',markersize=8)
# plt.plot(valid_acc_wo_aug_wo_crop, label='neither', color=colors[2], linewidth=2, marker='s',markersize=6)
plt.plot(valid_acc_ours, label='ours', color=colors[3], linewidth=3, marker='d', markersize=6)

plt.legend(loc='best', fontsize=16)

plt.title('Prediction Accuracy on Validation Set', fontsize=20)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Acc', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True, linewidth=1.8)
sns.despine()
ax.set_xticks(range(0, 15, 2))  # 原始刻度为 0, 2, 4, ..., 14
ax.set_xticklabels([str(i+1) for i in range(0, 15, 2)])
plt.savefig('acc_comparison.png', format='png', dpi=500, bbox_inches='tight')

plt.show()
