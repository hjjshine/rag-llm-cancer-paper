# utils/io.py
import pickle

def save_object(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, "rb") as input:
        obj = pickle.load(input)
    return(obj)
