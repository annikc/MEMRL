import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

sys.path.insert(0, '../modules/')
from basic.modules.Utils import running_mean as rm

import os
print('current dir ', os.getcwd())

data_dir = '../Data/results/'


oh = ['c4632834-1950-4487-8a15-14cf6dd4a330',
'73159540-face-484e-8372-49933a5b40bd',
'3712bc3e-75de-4e35-af75-8eed11f7f387',
'bc75125e-446b-43a6-89dc-a35c3003e6e6',
'a8d11e6f-63cc-417d-8d4a-a6429c2f52da']

pc_01 = [
'9f562faa-41dd-415f-afdb-33488309b170',
'86368121-9f72-44c7-a2d5-0b25f2039695',
'd598273d-c994-498d-86cb-9b7fad29aecf',
'93840aff-8148-41ac-a7e0-2d93cde1c820',
'9f8e85c4-c160-4662-9464-b661e510fa47'
]

pc_02 = ['15fc5ae4-20b8-453c-ac42-fa296e7b8754',
'8411546a-ddca-4045-a32f-575f4d5b9ad2',
'5947b44b-ba3d-42e5-8850-54e9b06a95a8',
'969d4b03-4285-4041-ae14-b230712a8dfa',
'0111aa72-4cab-4c0c-941a-12723c196c2c'
]

pc_03 = ['3dd4e397-88a0-478c-b9e0-e4b36553ad00',
'f6faf752-5e6e-48ca-9196-6229ca69c4c7',
'32733a64-5003-4ca4-8ec3-04f4b936d2de',
'888cbd85-6298-4df4-9658-ac359b197d42',
'd357f0a2-dd63-4e15-a225-25c78fb23c7a'
]

big_dict = {'oh':oh, 'pc_01':pc_01, 'pc_02':pc_02, 'pc_03':pc_03}

results_dict = {}
for key, value in big_dict.items():
    print(key)
    results_dict[key] = []
    results = []

    for id_num in value:
        print(id_num)
        file_addr = data_dir + f'{id_num}_data.p'
        print(f"--  loc:  {file_addr}")
        with open(file_addr, 'rb') as f:
            try:
                print(data_dir)
                dats = pickle.load(f)
            except:
                print("!!!!!! !!!!!  errors")
            reward_info = dats['total_reward']
            results.append(reward_info)

    pp = np.vstack(results)

    smoothing = 100
    avg_ = rm(np.mean(pp, axis=0),smoothing)[0:2000]
    std_ = rm(np.std(pp, axis=0), smoothing)[0:2000]
    results_dict[key].append(avg_)
    results_dict[key].append(std_)
