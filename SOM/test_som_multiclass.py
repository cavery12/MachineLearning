# Code that tests SOM implementation on sample multi-class data
# Chris Avery: April 26, 2021
import self_organizing_map as SOM
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load Some Data
data = datasets.load_wine();
traj = data.data;
labels = data.target;

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
map_size = 5;
input_size = traj_norm.shape[1];
update_radius = 2;
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
for epoch in range(30):
    print("Taining Epoch "+str(epoch+1)+" -----------------------");
    
    # Sweep through one epoch of training
    for i in range(X_train.shape[0]):
        som.update_weights(X_train[i]);
    
    valcounts = som.map2latent(X_val);
    print("----------------------------------------------- Done");
    print("")
    print("Validation Counts ----------------------------------");
    print(valcounts);
    print("----------------------------------------------------");
    print("");

# Test the model against true classifications
true_0 = traj_norm[np.where(labels==0)]
true_1 = traj_norm[np.where(labels==1)]
true_2 = traj_norm[np.where(labels==2)]

print("True Class 1")
print(som.map2latent(true_0))
print("True Class 2")
print(som.map2latent(true_1))
print("True Class 3")
print(som.map2latent(true_2))

labeled = (traj_norm[np.where(labels==0)],
           traj_norm[np.where(labels==1)], 
           traj_norm[np.where(labels==2)])
som.threeclass_plot(labeled);