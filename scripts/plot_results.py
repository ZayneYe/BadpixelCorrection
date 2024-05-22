import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

group_colors = sns.color_palette("pastel", 6)

# error_percentage = [0.40, 0.55, 0.70, 0.85]
# patch5_loss = [0.0565, 0.1709, 0.2617, 0.2545]
# patch9_loss = [0.0752, 0.1514, 0.2525, 0.3507]
# patch13_loss = [0.1310, 0.1194, 0.3050, 0.3899]
# # mae_loss = [_, 0.1855, 0.0983, _]
error_percentage = [0.01, 0.40, 0.70, 0.85]
patch5_loss = [0.005, 0.0565, 0.2617, 0.2545]
patch9_loss = [0.006, 0.0752, 0.2525, 0.3507]
patch13_loss = [0.007, 0.1310, 0.3050, 0.3899]
# mae_loss = [_, 0.1855, 0.0983, _]
# mae_loss = [0.053, 0.0826, 0.0983, 0.1891] # on normalized data
mae_loss = [0.049, 0.0894, 0.0902, 0.1519]

plt.figure(figsize=(8,7))
plt.plot(error_percentage, patch5_loss, label='MLP5x5', marker='*', markersize=20, linewidth=3.5, color=group_colors[0])
plt.plot(error_percentage, patch9_loss, label='MLP9x9', marker='*', markersize=20, linewidth=3.5, color=group_colors[1])
plt.plot(error_percentage, patch13_loss, label='MLP13x13', marker='*', markersize=20, linewidth=3.5, color=group_colors[2])
plt.plot(error_percentage, mae_loss, label='ViT AE', marker='*', markersize=20, linewidth=3.5, color=group_colors[3])
plt.xlabel('Error (%)', size=29)
plt.ylabel('NMSE', size=29)
plt.xticks(size=25)
plt.yticks(size=25)
legend=plt.legend(fontsize=24, loc='upper left', bbox_to_anchor=(0.02, 1.0), ncol=1)
legend.get_frame().set_alpha(0.5)
plt.tight_layout()
plt.grid(True, which='both', linestyle='--', linewidth=1.5)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, prop ={'size':18})
# plt.grid()
plt.savefig('mae_vs_mlp.png')


# MIT FiveK Dataset
patch5_loss = [0.0009, 0.0465, 0.3056, 0.2479]
patch9_loss = [0.0016, 0.0703, 0.3259, 0.4258]
patch13_loss = [0.0016, 0.1328, 0.3783, 0.4365]
mae_loss = [0.0036, 0.0070, 0.0155,0.0223]

plt.figure(figsize=(8,7))
plt.plot(error_percentage, patch5_loss, label='MLP5x5', marker='*', markersize=20, linewidth=3.5, color=group_colors[0])
plt.plot(error_percentage, patch9_loss, label='MLP9x9', marker='*', markersize=20, linewidth=3.5, color=group_colors[1])
plt.plot(error_percentage, patch13_loss, label='MLP13x13', marker='*', markersize=20, linewidth=3.5, color=group_colors[2])
plt.plot(error_percentage, mae_loss, label='ViT AE', marker='*', markersize=20, linewidth=3.5, color=group_colors[3])
plt.xlabel('Error (%)', size=29)
plt.ylabel('NMSE', size=29)
plt.xticks(size=25)
plt.yticks(size=25)
legend=plt.legend(fontsize=24, loc='upper left', bbox_to_anchor=(0.02, 1.0), ncol=1)
legend.get_frame().set_alpha(0.5)
plt.tight_layout()
plt.grid(True, which='both', linestyle='--', linewidth=1.5)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, prop ={'size':18})
# plt.grid()
plt.savefig('mae_vs_mlp_fivek.png')

# PSNR Comparison
corr_strategy = ['NNR', 'Linear', 'Median', 'ADC', 'Sparse', 'Ours']
psnr = [22.6, 25.7, 25.7, 23.5, 30.4, 30.55]
# c = ['red', 'yellow', 'black', 'blue', 'orange', 'green']
c = group_colors[:6]
# Sample data
x = np.arange(len(corr_strategy))  # the label locations

width = 0.35  # the width of the bars

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
rects1 = ax.bar(x, psnr, width, color=c)
ax.bar_label(rects1, size=14)
# rects3 = ax.bar(x + width/2, triplebn_acc, width, label='Triple BN', color='purple')
# ax.bar_label(rects3, padding=1)
# Adding labels and title
ax.set_ylabel('PSNR (dB)', size=20)
ax.set_xticks(x)
ax.set_xticklabels(corr_strategy, size=20)
# ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0),
        #   ncol=3, prop ={'size':15})
ax.yaxis.grid(True, which='both', linestyle='--', linewidth=1.5)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('SOTA_comparison.png')

# Bar Chart for comparison of Mean, Median, MLP
corr_strategy = ['Linear', 'Median', 'MLP']
nmse = [0.0783,	0.0887,	0.0055]
x = np.arange(len(corr_strategy))  # the label locations
c = ['red', 'y', 'black']
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
rects1 = ax.bar(x, nmse, width, color=c)
ax.bar_label(rects1, size=25)
ax.set_ylabel('NMSE', size=30)
ax.set_xticks(x)
ax.set_xticklabels(corr_strategy, size=25)
plt.yticks(fontsize=25)
plt.ylim(0, 0.1)
# ax.set_yticklabels(np.arange(0, 0.1, 0.5), size=50)
plt.tight_layout()
plt.savefig('mean_median_comparison.png')
