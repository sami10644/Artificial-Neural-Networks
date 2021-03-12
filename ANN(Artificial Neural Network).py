#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


import tensorflow as tf
from tensorflow import keras


# In[3]:


keras.__version__

tf.__version__
# In[4]:


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full,y_train_full),(X_test,y_test) = fashion_mnist.load_data()


# In[5]:


plt.imshow(X_train_full[4])


# In[6]:



class_names =["shirt/tops","Trouser","Pullover","Dress","Coat","sandal","shirt","sneaker","bag","ankle boot"]


# In[7]:


class_names[y_train_full[4]]


# In[8]:


class_names[y_train_full[1]]


# In[9]:


X_train_full[10]


# In[10]:


X_train_n = X_train_full /255
X_test_n = X_test /255 


# In[11]:


X_valid,X_train = X_train_n[:5000],X_train_n[5000:]
y_valid , y_train = y_train_full[:5000],y_train_full[5000:]
X_test = X_test_n


# In[12]:


X_valid[0]


# In[13]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation = "relu"))
model.add(keras.layers.Dense(100,activation = "relu"))
model.add(keras.layers.Dense(10,activation = "softmax"))


# In[14]:


model.summary()


# In[15]:


import pydot 
keras.utils.plot_model(model)


# In[16]:


weights,biases = model.layers[1].get_weights()


# In[17]:


weights


# In[18]:


weights.shape


# In[19]:


biases


# In[20]:


biases.shape


# In[21]:


model.compile(loss = "sparse_categorical_crossentropy",optimizer = "sgd",metrics = ["accuracy"])


# In[22]:


model_history = model.fit(X_train,y_train,epochs = 30,
                          validation_data=(X_valid,y_valid))


# In[23]:


model_history.params


# In[24]:


model_history.history


# In[25]:


import pandas as pd

pd.DataFrame(model_history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# In[26]:


model.evaluate(X_test,y_test)


# In[27]:


X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)


# In[28]:


y_pred = model.predict_classes(X_new)
y_pred


# In[29]:


np.array(class_names)[y_pred]


# In[93]:


print(plt.imshow(X_test[0]))


# In[31]:


print(plt.imshow(X_test[1]))


# In[32]:


print(plt.imshow(X_test[2]))


# Restoring Models and using Callbacks

# In[37]:


model.save("my_Func_model.h5")


# In[39]:


get_ipython().run_line_magic('pwd', '')


# In[65]:


del model


# In[66]:


keras.backend.clear_session()


# In[89]:


model = keras.models.load_model("my_Func_model.h5")


# In[90]:


model.summary()


# In[82]:


y_pred = model.predict(X_new)
print(y_pred)


# In[141]:


del model


# In[142]:


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


# In[147]:


model = keras.models.Sequential([
    keras.layers.Dense(30,activation = "relu",input_shape=[8]),
    keras.layers.Dense(30,activation = "relu"),
    keras.layers.Dense(1)
])


# In[148]:


model.compile(loss ="mse" ,optimizer = keras.optimizers.SGD(lr=1e-3))


# In[149]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("Model-{epoch:02D}.h5")


# In[107]:



history = model.fit(X_train,y_train,epochs = 10,
                   validation_data=(X_valid,y_valid
                    callbacks = [checkpoint_cb])


# In[ ]:




