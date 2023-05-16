import pickle
import numpy as np

with open("results/23-04-23@12:55:53/mbpp_sanitized_with_bert_encodings.pickle", "rb") as file:
    data = pickle.load(file)

BERT_DIMENSION = 768

embeddings = np.array([row["bert_embedding"].detach().numpy() for row in data])
assert embeddings.shape == (len(data), BERT_DIMENSION)

dot_products = embeddings.dot(embeddings.T)
assert dot_products.shape == (len(data), len(data))

norms = np.linalg.norm(dot_products, 2, axis=1)
assert norms.shape == (len(data),)

norm_products = norms[:, None].dot(norms[None, :])
assert norm_products.shape == (len(data), len(data))

cosines = dot_products / norms
assert cosines.shape == (len(data), len(data))

test_index = 3
top = np.argsort(
    cosines[test_index]
)

print(data[test_index]["prompt"])
print("-" * 50)
for i in top[-4:-1]:
    print(data[i]["prompt"])
    print()

# row1 = data[argmax[0]]
# row2 = data[argmax[1]]
# print()
# print(row1["prompt"])
# print()
# print(row2["prompt"])