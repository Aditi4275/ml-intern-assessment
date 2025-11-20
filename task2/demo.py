import numpy as np
from scaled_dot_product_attention import scaled_dot_product_attention


np.random.seed(142)

# Let's use 2 queries, 4 keys, 4 values, d_k = d_v = 3
Q = np.random.rand(2, 3)
K = np.random.rand(4, 3)
V = np.random.rand(4, 3)

print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)

output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("\nAttention weights:\n", attention_weights)
print("\nAttended output:\n", output)
