# Code that tests SOM implementation on BL data
# Chris Avery: April 26, 2021
import self_organizing_map as SOM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_one_bl_traj(path, filename):
    traj = np.loadtxt(fname=path+filename);
    return traj.T;

# Load Some Data
path = "C:\\Users\\csa97\\Programming\\Python\\MachineLearning\\data\\betalac_AC\\"
traj1 = load_one_bl_traj(path, "1ERM_TEM1_AlphaCarbon_GLOB.txt");
traj2 = load_one_bl_traj(path, "1ERM_TEM52_AlphaCarbon_GLOB.txt");
traj3 = load_one_bl_traj(path, "1ERM_TEM2_AlphaCarbon_GLOB.txt");
traj = np.concatenate((traj1, traj2, traj3), axis=0);

# Preprocess Some Data
scaletype = "standard"
if scaletype == "minmax":
    scaler = MinMaxScaler().fit(traj);
elif scaletype == "standard":
    scaler = StandardScaler().fit(traj);

traj_norm = scaler.transform(traj);

# Split into Train, Test, Validation
X_train, X_test = train_test_split(traj_norm, train_size=.8);
X_train, X_val =  train_test_split(X_train, train_size=.75);

# Define SOM model
map_size = 7;
input_size = traj_norm.shape[1];
update_radius = 3;
learning_rate = .1;
som = SOM.SOM_Model(map_size, input_size, update_radius, learning_rate, scaletype);

#out = som.forward(traj_norm[0]);
#bmu = som.find_bmu(out);
#neighborhood = som.find_bmu_neighborhood(bmu);
#print(out)
#print(bmu)
#print(neighborhood)

## Train Model
print("Pretraining Counts: ");
valcounts = som.map2latent(X_val);
print(valcounts)
print("");
for epoch in range(10):
    print("Taining Epoch "+str(epoch+1)+" -----------------------");
    
    # Sweep through one epoch of training
    for i in range(int(np.floor(X_train.shape[0]/2))):
        som.update_weights(X_train[i]);
    
    valcounts = som.map2latent(X_val);
    print("----------------------------------------------- Done");
    print("")
    print("Validation Counts ----------------------------------");
    print(valcounts);
    print("----------------------------------------------------");
    print("");
    #time.sleep(10);

som.threeclass_plot((traj1, traj3, traj2))