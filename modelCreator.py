# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:13:05 2018

@author: PascPeli
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 01:05:48 2018

@author: PascPeli
"""

from keras.models import Model
from keras.layers import (Input, Conv2D, BatchNormalization, Activation, Dropout,
                          Dense, Flatten, MaxPooling2D, ZeroPadding2D, concatenate)
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, EarlyStopping


class Model_creator ():
    def __init__(self, cnn_params, dense_params, input_size=(100,100,3), optimizer='adam',
                 loss='categorical_crossentropy', metrics=['accuracy'], callbacks=[]):
        #self.nof_classes = nof_classes
        self.input_size = input_size
        self.cnn_params = cnn_params
        self.dense_params = dense_params
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        
        
    def CNN_layer(self, X, filters, kernel=(3, 3), stride=(2, 2), padding='valid', activation='relu', kernel_initializer='glorot_uniform', mp_kernel=None, mp_stride=None, name = ''):
        '''
        Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', activation=None, kernel_initializer='glorot_uniform')
        '''    
        if mp_kernel==None: mp_kernel=kernel
        
        if mp_stride==None: mp_stride=stride
        
        X = Conv2D(filters, kernel_size=kernel, strides = stride, padding='valid', name = name, kernel_initializer = kernel_initializer)(X)
        X = BatchNormalization(axis = 3, name = name+'_BatchNorm')(X)
        X = Activation(activation)(X)
        X = MaxPooling2D(mp_kernel, strides=mp_stride, name=name+'_MaxP')(X)
        
        return X
    
    def Dense_layer(self, X, units, activation, kernel_initializer, dropout, name):
        if len(name)==2:
            X = Dense (units=units, activation=activation, kernel_initializer=kernel_initializer, name=name[0])(X)
            X = Dropout(dropout, name=name[1])(X)
        else:
            X = Dense (units=units, activation=activation, kernel_initializer=kernel_initializer, name=name)(X)
        return X
    
    
    def create_CNN_Model(self):
        
        X_in = Input(self.input_size, name='Input_layer')
        
        X = ZeroPadding2D((2, 2)) (X_in)
        
        i=0
        for (filters, kernel, stride, padding, activation, initializer, mp_kernel, mp_stride, name) in self.cnn_params:
            if (not name):
                i=+1
                name = 'CNN_layer_'+str(i)
            X = self.CNN_layer(X,filters, kernel, stride, padding, activation, initializer, mp_kernel, mp_stride, name)
        i=0
        X = Flatten(name='Flatten_layer')(X)
        for (units, activation, kernel_initializer, dropout, name) in self.dense_params[:-1]:
            if (not name):
                i=+1
                name = ['Dense_layer_'+str(i), 'Dropout_'+str(i)]
            else:
                name = [name, name+'_Dropout']
            X = self.Dense_layer(X, units, activation, kernel_initializer, dropout, name)
        
        self.nof_classes, activation, kernel_initializer, dropout, name = self.dense_params[-1]
        X_out = self.Dense_layer (X, self.nof_classes, activation, kernel_initializer, dropout, name)
        #X_out = Dense (self.nof_classes, activation='softmax', name='output', kernel_initializer = glorot_uniform(seed=0))(X)
        self.CNN_model = Model(inputs=X_in, outputs=X_out)
        
    def compile_Model(self, callbacks):
        if hasattr(self,'model'):
        #try:
            if callbacks:
                checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, 
                                            save_best_only=True, save_weights_only=False, mode='auto', period=1)
                early = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=0, mode='auto')
                self.callbacks_list = [checkpoint,early]
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        else:
        #except AttributeError:
            print('You havent created a Model yet. Run create_Model()')
    
    
    def save_Model(self):
        '''
        '''
        if hasattr(self,'model'):
            #saving the weights
            self.CNN_model.save_weights("weights.hdf5",overwrite=True)
            
            #saving the model itself in json format:
            model_json = self.CNN_model.to_json()
            with open("model.json", "w") as model_file:
                model_file.write(model_json)
            print("Model has been saved.")
        else:
        #except AttributeError:
            print('You havent created a Model yet. Run create_Model()')

    
