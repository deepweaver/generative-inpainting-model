import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
with open("./logs.log", 'r') as file:
    content = file.readlines()

sns.set()
pal = sns.color_palette("Set1")

max_iter = len(content)
x = range(0,329700,100)
print(len(x)) 
print(max_iter)
y = [] 
print("saving at {}".format(max_iter))
print(max_iter)
for i in range(0,max_iter):
    line_list = content[i].split(' ')
    loss_idx = line_list.index('l1:')
    loss_val = line_list[loss_idx+1]
    # print(float(loss_val)) 
    y.append(float(loss_val) )
plt.plot(x, y, linewidth = '1', color='cornflowerblue', label='l1', alpha=0.8) 
plt.legend(loc='upper right')
# plt.show() 

# y = np.vstack([y_a, y_b, y_c])
# labels = ['a', 'ab', 'b']
# plt.ylim([0,32])
# plt.stackplot(x, y_a, y_b, y_c, labels=labels,colors=pal, alpha=0.4 )



# plt.legend(loc='upper left')

plt.savefig("loss_{}.png".format(max_iter),dpi=500)
