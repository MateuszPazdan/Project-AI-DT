
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import hickle as hkl
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


x,y_t,x_norm,x_n_s,y_t_s = hkl.load('haberman.hkl')
y_t -= 1
x=x.T
y_t = np.squeeze(y_t)
print(y_t.shape, x.shape)

criterion_vec = ["gini", "entropy", "log_loss"]
criterion_num_vec = np.array(range(len(criterion_vec)))
for ind_criterion in range(len(criterion_vec)):
    print(criterion_vec[ind_criterion])

max_depth_vec = np.array(range(2,11, 2))

data = x
target = y_t

CVN = 10
skfold = StratifiedKFold(n_splits=CVN)

PK_cr_md_vec = np.zeros([len(criterion_vec),len(max_depth_vec)])

for criterion_ind in range(len(criterion_vec)):
    for max_depth_ind in range(len(max_depth_vec)):
        PK_vec = np.zeros(CVN)
        for i, (train, test) in enumerate(skfold.split(data, target), start=0):
            x_train, x_test = data[train], data[test]
            y_train, y_test = target[train], target[test]
            # print(i,train,test)
            decision_tree = DecisionTreeClassifier(random_state=0, \
            max_depth=max_depth_vec[max_depth_ind], \
            criterion=criterion_vec[criterion_ind])
            decision_tree = decision_tree.fit(x_train, y_train)
            result = decision_tree.predict(x_test)

            n_test_samples = test.size
            PK_vec[i] = np.sum(result == y_test) / n_test_samples

        PK_cr_md_vec[criterion_ind,max_depth_ind] = np.mean(PK_vec)
        print("criterion: {} | max_depth: {} | PK: {}".format(criterion_vec[criterion_ind],\
                                                    max_depth_vec[max_depth_ind],\
                                                    PK_cr_md_vec[criterion_ind,max_depth_ind]))
        
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(criterion_num_vec, max_depth_vec)
surf = ax.plot_surface(X, Y, PK_cr_md_vec.T, cmap='viridis')

ax.set_xlabel('criterion')
ax.set_xticks(np.array(range(len(criterion_vec))))
ax.set_xticklabels(criterion_vec)
ax.set_ylabel('max_depth')
ax.set_zlabel('PK')

ax.view_init(30, 200)
plt.show()
plt.savefig("Fig.1_dtree_CV_experiment.png",bbox_inches='tight')