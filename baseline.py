# this code is used to
# get xvector.ark from xvector.npz and
# calculate baseline EER

import numpy as np
import tensorflow as tf
import os


paths = ["./data/voxceleb_combined_200000/xvector",
         "./data/sitw_dev/enroll/xvector",
         "./data/sitw_dev/test/xvector",
         "./data/sitw_eval/enroll/xvector",
         "./data/sitw_eval/test/xvector"
         ]

# delete
for path in paths:
    if os.path.exists(path+'.ark') == True:
        os.remove(path+'.ark')
        print('delete {}.ark'.format(path))

# write
for path in paths:
    # load npz data
    vector = np.load(path+'.npz')['vector']
    labels = np.load(path+'.npz')['label']
    with open(path+'.ark', 'w') as f:
        for i in range(vector.shape[0]):
            f.write(str(labels[i]))
            f.write('  [ ')
            for j in vector[i]:
                f.write(str(j))
                f.write(' ')
            f.write(']')
            f.write('\n')
    print('{}.ark is done!'.format(path))
    
print('\nall done!')
