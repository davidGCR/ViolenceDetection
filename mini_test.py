import torch
import numpy as np

outs = torch.tensor([[-1.6251, 1.2137], [-0.7444, 0.7942], [-2.1056, -0.0733], [-1.1363, 1.3683]], dtype=float)
outs = outs.numpy()
max_values = np.max(outs)
min_values = np.min(outs)
# min_values = min_values.reshape([4, 1])
# max_values = max_values.reshape([4,1])
# print(outs, outs.size())
# print(min_values, min_values.size())
score = (outs - min_values)/(max_values-min_values)

print(score, score.shape)


# print(score)