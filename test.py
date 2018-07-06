import numpy as np

ting = []
ting.append(np.arange(9))
ting.append(np.arange(3)*7)
ting.append(np.arange(2))
ting=np.array(ting)

jan = []
jan.append(ting)
jan.append(ting)
jan = np.array(jan)

print(jan.shape)
print(jan)
print(np.dot(np.array(jan[:,-1]),np.ones(2)))