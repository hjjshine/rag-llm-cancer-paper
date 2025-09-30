# utils/io.py
import pickle
import random
import math

def save_object(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, "rb") as input:
        obj = pickle.load(input)
    return(obj)

def split_ranges(max_int, num_ranges, samples=None, seed=42):
    random.seed(seed)
    step = math.ceil(max_int / num_ranges)
    ranges = [(i*step+1, min((i+1)*step, max_int)) for i in range(num_ranges)]
    
    if samples is None:
        return ranges
    
    # split samples evenly across ranges
    n_per_range = samples // num_ranges
    remainder = samples % num_ranges
    
    result = []
    for i, r in enumerate(ranges):
        count = n_per_range + (1 if i < remainder else 0)
        result.extend(random.randint(r[0], r[1]) for _ in range(count))
    
    return result