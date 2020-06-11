import os
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split

# Change Working Directory to data_dir
data_dir = '../Data/Pradeep_16/'
os.chdir(data_dir)

# Load .wav Files into arrays
data_x = []
data_y = []
for a,i in enumerate(['forward','back','left','right','stop']):
    lis = os.listdir(data_dir+i)
    for j in lis:
        l,sr = sf.read(data_dir+i+'/'+j)
        if(len(l)==16000):
            data_x.append(l)
            data_y.append(a)
        else:
            print(i)
    print(i," Done")
data_x = np.array(data_x)
data_y = np.array(data_y)

# Split 20% as test data. Random_state is used to reproduce the split
# Stratify to maintain equal proportions of classes
tr_x,te_x, tr_y,te_y = train_test_split(data_x,data_y,stratify=data_y,random_state=123,test_size=0.2)

# Augment Train Data by time shifting in 25000 length array of zeros
x_train = []
y_train = []
for i,j in enumerate(tr_x):
    x= len(j)
    p = 25000-x
    for y in range(1 ,p, 500):
        nx = np.zeros(25000)
        # Fill the audio
        nx[y:y+x] =j
        x_train.append(nx)
        y_train.append(tr_y[i])
x_train = np.array(x_train)
y_train = np.array(y_train)

# Save the augmented train set
np.save('x_train.npy',x_train)
np.save('y_train.npy',y_train)

# Augment Train Data by time shifting in 25000 length array of zeros
x_test = []
y_test = []
for i,j in enumerate(te_x):
    x= len(j)
    p = 25000-x
    for y in range(1 ,p, 500):
        nx = np.zeros(25000)
        # Fill the auido
        nx[y:y+x] =j
        x_test.append(nx)
        y_test.append(te_y[i])
x_test = np.array(x_test)
y_test = np.array(y_test)

# Save the augmented train set
np.save('x_test.npy',x_test)
np.save('y_test.npy',y_test)