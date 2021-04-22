import matplotlib.pyplot as plt

plt.title("Evaluation Precision")
plt.ylabel('Percent %')
plt.xlabel('Epoch')
plt.plot([1, 2, 3, 4,5,6,7,8,9,10], [44.305, 92.099, 97.704, 98.953, 99.352, 99.678, 99.732, 99.812, 99.873, 99.925], color="green", label="top1 train")
plt.plot([1, 2, 3, 4,5,6,7,8,9,10], [77.585, 99.699, 99.953, 99.991, 99.998, 99.998, 100.00, 100.00, 99.998, 100.00], color="blue", label="top5 train")
plt.plot([1, 2, 3, 4,5,6,7,8,9,10], [66.437, 79.660, 86.239, 88.907, 91.081, 90.457, 91.584, 91.053, 93.413, 90.475], color="red", label="top1 test")
plt.plot([1, 2, 3, 4,5,6,7,8,9,10], [94.295, 97.379, 98.654, 98.812, 99.393, 98.841, 99.446, 98.899, 99.375, 99.121], color="orange", label="top5 test")

legend = plt.legend(loc='lower right')

plt.show()
