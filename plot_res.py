
import numpy as np
import matplotlib.pyplot as plt

train_loss_total_his = np.loadtxt('result/train_loss_total_his.txt')
test_loss_total_his = np.loadtxt('result/test_loss_total_his.txt')

plt.plot(train_loss_total_his)
plt.title('train_loss_total_his')
plt.show()
plt.plot(test_loss_total_his)
plt.title('test_loss_total_his')
plt.show()