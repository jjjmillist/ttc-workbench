import pickle
import numpy as np
    
with open("results/perplexity-28-04-23@09:37:59/perplexities", "rb") as file:
    data = pickle.load(file)

perplexities = data["perplexities"]
inputs = data["inputs"]

n_data = int(np.sqrt(len(perplexities)))
assert len(perplexities) == n_data**2

offset = 0
P = np.empty((n_data, n_data))
X = np.empty((n_data, n_data), dtype=str)
for prefix_index in range(n_data):
    for test_index in range(n_data):
        P[test_index, prefix_index] = perplexities[offset]
        X[test_index, prefix_index] = inputs[offset]
        offset += 1

# for test_index in range(n_data):
#     sorted_indices = np.argsort(P[test_index])
#     print(P[test_index])

print(np.argsort(P, axis=0))
print("----")
print(np.argsort(P, axis=1))
print("----")
print(np.argsort(P[0, :10]))
print("----")
print(np.argsort(P[:10, 0]))