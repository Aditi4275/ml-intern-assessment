import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    '''
    Arguments:
     Q: (seq_len_q, d_k)
     K: (seq_len_k, d_k)
     V: (seq_len_k, d_v)
     mask: (seq_len_q, seq_len_k)

    Outputs:
     output: (seq_len_q, d_v)
     attention_weights: (seq_len_q, seq_len_k)
    '''
    
    
    d_k = Q.shape[-1]
    # 1.scores 
    scores = Q @ K.T
    
    # 2. Scale 
    scores = scores / np.sqrt(d_k)

    # 3. Masking
    if mask is not None:
        scores = np.where(mask, -1e9, scores)  # -1e9 forces softmax ~0

    # 4. Softmax over the last axis (keys) for each query position
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 5. output = attention_weights
    output = attention_weights @ V 

    return output, attention_weights
