from time import sleep
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

def fun(x):
    sleep(int(x))
    return f'x={x}'


with ThreadPoolExecutor() as pe:
    iter = pe.map(fun, [1,2,3,4,5])
    for _ in tqdm(iter):
        pass
