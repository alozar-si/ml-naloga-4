#%%
from scipy.integrate import ode
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers
from sklearn.preprocessing import MinMaxScaler
from time import time
#%%
def lorenz(t, y, params):
    """
    Diferencialne enacbe Lorentzovega modela
    y = [x,y,z]
    dy = [dx/dt, dy/dt, dz/dt]
    params=[sigma,rho,beta]
    """
    dy=np.zeros(3)
    dy[0]=params[0]*(y[1]-y[0])
    dy[1]=(params[1]-y[2])*y[0] - y[1]
    dy[2]=y[0]*y[1] - params[2]*y[2]
    return dy

### Integracija Lorenzovega sistema v času ###

# definiarmo vrenosti parametrov params=[sigma,rho,beta]
params = [10., 28.,8./3.]

#nastavimo zacetni cas na 0
t = 0.
#dolzina casovnega koraka
dt=0.01

#koncni cas integracije
tmax=10000.

#Dolocimo vrednosti spremenljivk x,y, ob casu 0;
#nakljucne vrednosti so zrebane po enakomerni porazdelitvi
y0_ATM = np.array([np.random.uniform(-20,20),np.random.uniform(-20,20),np.random.uniform(10,30)])

# nasa njboljsa ocena trenutnega stanja ozracja ima napako
c=0.2
y0_MOD=y0_ATM + c*np.random.random(3)

#nastavimo integrator
solver_ATM=ode(lorenz).set_integrator('dopri5')
solver_ATM.set_f_params(params)
solver_ATM.set_initial_value(y0_ATM, t)

solver_MOD=ode(lorenz).set_integrator('dopri5')
solver_MOD.set_f_params(params)
solver_MOD.set_initial_value(y0_ATM, t)

#%% integrate
n = int(tmax/dt) + 1
data = np.zeros((n, 3))

i = 0
start = time()
while t < tmax:
    y_ATM = solver_ATM.integrate(t+dt)
    #y0_MOD = solver_MOD.integrate(t+dt)
    data[i, :] = y_ATM[:]
    i = i + 1
    t += dt
print("Elasped time: %0.3g s" %(time()-start))
data = data[1000:] # only use data from the attractor (after time = 10000 * dt)
print("Integration finished.")

#%% NN

data = np.load("data.npy")
#PREPROCESS DATA
x_train = data[:-1,:]
y_train = data[1:, :] - data[:-1, :]

x_scaler=MinMaxScaler()
y_scaler=MinMaxScaler()

x_scaler.fit(x_train)
y_scaler.fit(y_train)

x_train_norm=x_scaler.transform(x_train)
y_train_norm=y_scaler.transform(y_train)

xy_train_norm=np.concatenate((x_train_norm,y_train_norm), axis=1)

new_order=np.random.choice(range(x_train.shape[0]),x_train.shape[0],replace=False)
x_train_norm=x_train_norm[new_order]
y_train_norm=y_train_norm[new_order]
# %%
from tensorflow.keras.layers import Dense, Dropout
# create a "Sequential" model and add a Dense layer as the first layer
# ce nasa Loss ne vec pade se ustavi minimizacija
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')

#initializer = 
#ce dosezemo plato zmanjsamo korak ucenja
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

model = tf.keras.models.Sequential()
model.add(Dense(3, input_dim=3, activation='relu'))
model.add(Dense(81, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(81, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(81, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(81, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['mae'])
# %%
# Train model on data
start=time()
history = model.fit(x_train_norm, y_train_norm, epochs=200, batch_size=1024,validation_split=0.2,
                    callbacks=[callback,reduce_lr])
print("Elasped time: %0.3g s" %(time()-start))
print(model.summary())

# %%
