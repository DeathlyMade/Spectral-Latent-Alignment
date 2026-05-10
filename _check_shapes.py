import numpy as np
files = [
    'ratings_Office_Products.csv',
    'ratings_Movies_and_TV.csv',
    'ratings_Sports_and_Outdoors.csv',
    'ratings_CDs_and_Vinyl.csv',
    'ratings_Apps_for_Android.csv',
    'ratings_Video_Games.csv',
    'ratings_Toys_and_Games.csv',
    'ratings_Automotive.csv',
]
for f in files:
    arr = np.load(f'Data/{f}.npy')
    print(f'{f}: {arr.shape}')
