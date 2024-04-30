from time import sleep
from tqdm import tqdm

for i in tqdm(range(100), desc="Loading..."):
    sleep(2)
    pass