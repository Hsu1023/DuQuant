import numpy as np
import torch
from tqdm import tqdm
import pickle

device = 'cuda'

for n in [int(2**i) for i in range(1, 13)]:

    v_i = 1 / np.sqrt(n)

    R = torch.eye(n).to(device)

    def exchange(i, j, tensor):
        indices_row = torch.arange(tensor.size(0))
        indices_row[i], indices_row[j] = indices_row[j].item(), indices_row[i].item()
        tensor = tensor[indices_row]

        indices_col = torch.arange(tensor.size(1))
        indices_col[i], indices_col[j] = indices_col[j].item(), indices_col[i].item()
        tensor = tensor[:, indices_col]
        return tensor


    for i in tqdm(range(n-1)):
        rot_i = torch.eye(n).to(device)
        cos = 1 / np.sqrt(i+2)
        sin = np.sqrt(i+1) / np.sqrt(i+2)
        rot_i[i, i] = rot_i[i + 1, i + 1] = cos
        rot_i[i, i + 1] = -sin
        rot_i[i + 1, i] = sin
        R = torch.matmul(rot_i, R)

    R = exchange(0, n-1, R).cpu()
    try:
        dic = pickle.load(open('Rot.pkl', 'rb'))
    except:
        dic = {}
    dic.update({n: R})

    pickle.dump(dic, open('Rot.pkl', 'wb'))