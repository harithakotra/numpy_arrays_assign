import numpy as np
np.random.seed(42)
scores = np.random.randint(50, 101, size=(5, 4))
print("Scores:\n", scores)
print(f"\nScore of 3rd student in 2nd subject: {scores[2, 1]}")
print(f"\nScores of last 2 students: \n{scores[-2:]}")
print(f"\nFirst 3 students, subjects 2 and 3:\n{scores[:3, 1:3]}")
# -----------
# Task 2
# -----------
col_mean = np.round(scores.mean(axis=0), 2)
print(f"\nColumn-wise mean: {col_mean}")
curve = np.array([5, 3, 7, 2])
curved_scores = scores + curve
curved_scores = np.clip(curved_scores, None, 100)
print(f"\n Curved Scores :\n{curved_scores}")
row_max = curved_scores.max(axis=1)
print(f"\nRow-wise max: {row_max}")
# ---------
# Task 3
# ---------
row_min = curved_scores.min(axis=1, keepdims=True)
row_max = curved_scores.max(axis=1, keepdims=True)
normalized = (curved_scores - row_min) / (row_max - row_min)
print(f"\nNormalized Scores:\n{normalized}")
max_index = np.unravel_index(np.argmax(normalized), normalized.shape)
print(f"\nHighest normalized value at (student_index, subject_index): {max_index}")
above_90 = curved_scores[curved_scores > 90]
print(f"\nScored above 90: {above_90}")