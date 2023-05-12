from matplotlib import pyplot as plt


train_times = [1801, 19441, 83177]
patch_sizes = [5, 9, 14]
plt.figure(figsize=(12,6))
plt.title('Total amount of model parameters of different patches', size=15)
plt.xlabel('Number of model parameters', size=15)
plt.ylabel('Patch Size', size=15)
plt.yticks(patch_sizes, size=15)
plt.xticks([20000, 40000, 60000, 80000], size=15)
for x, y in zip(train_times, patch_sizes):
    plt.text(x, y, '%d' % x)
plt.barh(patch_sizes, train_times, height=2)
plt.savefig('Model_parameters.png')
