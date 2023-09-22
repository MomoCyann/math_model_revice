import matplotlib.pyplot as plt

plt.figure(figsize=(6,3))

plt.xlabel("Number of Clusters")
plt.ylabel("BIC Index")
plt.rc('font', family='Times New Roman')
plt.subplots_adjust(bottom=0.15)
plt.savefig('./fig/bic_cluster.tif', dpi=600)
plt.show()
