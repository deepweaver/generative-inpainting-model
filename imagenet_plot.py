import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
with open("./imagenet_all_iter_answers.txt", 'r') as file: 
    content = file.readlines()
pal = sns.color_palette("Set1")
# print(content[3], end='')
# print(len(content))
x = [] 
y = [] 
y_a = [] 
y_b = [] 
y_c = [] 
max_iter = (len(content)-1)//4
print(max_iter)
for i in range(0,max_iter,4):
    # print(,end='')
    x.append(int(content[i].split(" ")[-1]))
    dt = list(map(int,content[i+2].split(",")[:3]))
    y_a.append(dt[0]) 
    y_b.append(dt[1]) 
    y_c.append(dt[2]) 
    # y.append(sum(map(int,content[i+2].split(",")[:3])))
y = np.vstack([y_a, y_b, y_c]) 
labels = ['a', 'ab', 'b'] 

# labels = ["Fibonacci ", "Evens", "Odds"]
plt.ylim([0,32])
plt.stackplot(x, y_a, y_b, y_c, labels=labels,colors=pal, alpha=0.4 ) 
plt.legend(loc='upper right')



# fig, ax = plt.subplots()
# ax.set_ylim([0,32])
# # ax.stackplot(x, y)
# ax.stackplot(x, y_a, y_b, y_c, labels=labels,colors=pal, alpha=0.4 )
# ax.legend(loc='upper left')


# fig, ax = plt.subplots()
# ax.set_ylim([0,32])
# ax.stackplot(x, y)
# plt.show()
plt.savefig("cpm_{}.png".format(max_iter))