import os
import pickle
import torch

import sys
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import RetargetingProblem

num_var = 19
num_ineq = 38
num_eq = 0
num_examples = 1836

torch.set_default_dtype(torch.float64)
filepath = './datasets/Retargeting_dataset_var19_ineq38_eq0_ex1836_v2'

problem = RetargetingProblem(filepath)

problem._X = problem.X[:num_examples, :]
# print(problem._X)
problem._R_root = problem.R_root[:num_examples, :]
problem._R_root_trans = problem.R_root_trans[:num_examples, :]
problem._Y = problem.Y[:num_examples, :]

with open("./Retargeting_dataset_var{}_ineq{}_eq{}_ex{}_v2".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(problem, f)







# # Cut down number of samples if needed
# problem._X = problem.X[:num, :]
# problem._Y = problem.Y[:num, :]
# problem._num =  problem.X.shape[0]
#
# with open("./acopf{}_dataset".format(nbus), 'wb') as f:
#     pickle.dump(problem, f)

# with open(filepath, 'rb') as f:
#     data = pickle.load(f)
# for attr in dir(data):
#     var = getattr(data, attr)
#     if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
#         try:
#             setattr(data, attr, var.to(DEVICE))
#         except AttributeError:
#             pass
# data._device = DEVICE

# # 加载 pickle 文件
# with open(filepath, 'rb') as f:
#     data = pickle.load(f)
#
# # 提取所有张量属性并保存为字典
# tensors_dict = {}
# for attr in dir(data):
#     var = getattr(data, attr)
#     if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
#         tensors_dict[attr] = var
#
# # 保存张量字典为文件
# # 保存张量字典为 .pt 文件
# tensors_filepath = filepath.replace('.pkl', '_tensors.pt')  # 修改文件名
# torch.save(tensors_dict, tensors_filepath)
#
# print(f"All tensor attributes saved to {tensors_filepath}.")

