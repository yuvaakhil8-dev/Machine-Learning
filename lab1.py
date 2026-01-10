import numpy as np

# Q1:
def count_pair_sum(arr, target=10):
    pair_total = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] + arr[j] == target:
                pair_total += 1
    return pair_total

list_q1 = [2, 7, 4, 1, 3, 6]
print("Pairs count:", count_pair_sum(list_q1))


# Q2: 
def find_range(nums):
    if len(nums) < 3:
        return "Range determination not possible"
    max_val = max(nums)
    min_val = min(nums)
    return max_val - min_val

list_q2 = [5, 3, 8, 1, 0, 4]
print("Range:", find_range(list_q2))


# Q3: 
def compute_matrix_power(mat_data, exp):
    matrix_np = np.array(mat_data)
    return np.linalg.matrix_power(matrix_np, exp)

matrix_data = [[1, 2], [3, 4]]
print("Matrix power:\n", compute_matrix_power(matrix_data, 2))


# Q4: 
def highest_frequency_char(text_val):
    char_count = {}
    top_char = None
    top_count = 0

    for ch in text_val:
        if ch.isalpha():
            char_count[ch] = char_count.get(ch, 0) + 1
            if char_count[ch] > top_count:
                top_count = char_count[ch]
                top_char = ch

    return top_char, top_count

text_data = "hippopotamus"
char, freq = highest_frequency_char(text_data)
print(f"Q4 Most frequent character: {char}, Count: {freq}")


# Q5: 
random_gen = np.random.default_rng(42)
sample_data = random_gen.integers(1, 10, size=25)

def calculate_stats(values):
    mean_val = np.mean(values)
    median_val = np.median(values)

    highest_freq = 0
    mode_val = values[0]

    for v in values:
        count_v = list(values).count(v)
        if count_v > highest_freq:
            highest_freq = count_v
            mode_val = v

    return mean_val, median_val, mode_val

mean_val, median_val, mode_val = calculate_stats(sample_data)

print("Random data:", sample_data)
print("Mean:", mean_val)
print("Median:", median_val)
print("Mode:", mode_val)
