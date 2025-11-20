# Scaled Dot-Product Attention Implementation

## The Five Core Operations:

**1. Dot Product of Q and K Transpose:**
The similarity between each query and key is calculated as their dot product. This forms the raw attention scores.

**2. Scaling:**
To prevent the dot products from growing too large in magnitude (which can negatively impact training with softmax), the scores are scaled by dividing by square root of dimension of the keys.

**3. Masking (Optional):**
Positions can be masked by assigning a large negative value so that their influence is minimized after applying the softmax.

**4. Softmax Function:**
Transforms raw scores into normalized attention weights, indicating the importance of each key relative to the query.

**5. Weighted Sum of Values:**
The final output is computed as the weighted sum of value vectors weighted by the attention scores.

## Sample input
```python
Q:
 [[0.90206152 0.55780754 0.65598471]
 [0.83247141 0.19988419 0.12725426]]
K:
 [[0.77143911 0.43228855 0.38528223]
 [0.78364337 0.78553162 0.16280861]
 [0.11511333 0.33252155 0.60157345]
 [0.87780234 0.61487722 0.4240039 ]]
V:
 [[0.95784627 0.85408447 0.77414528]
 [0.87562841 0.07835524 0.03195872]
 [0.70624246 0.55297016 0.85526749]
 [0.95410747 0.47497613 0.5481674 ]]
```
## Sample output
```python
Attention weights:
 [[0.25505068 0.26436276 0.19045783 0.29012873]
 [0.26032612 0.26833093 0.19073107 0.28061187]]

Attended output:
 [[0.88710628 0.48167075 0.52782648]
 [0.88674686 0.48211816 0.52705412]]
```