import torch
import numpy as np
path = '/home/iccd/Tianxin/AAAI/SID/trainInfo.npy'
file_ = np.load(path)
print((file_[8]))
# for i in range(len(file_)):
#     e = torch.from_numpy(file_[i]['classLabel'])
#     if (len(torch.nonzero(e)) > 1):
#         print(file_[i]['imgName'])