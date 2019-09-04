#######################################################################################################
#####                       IMPORTACIÓN DE LIBRERÍAS Y RUTINASA UTILIZAR                          #####
#######################################################################################################

import numpy as np
import tensorflow as tf
import keras
from keras import regularizers, optimizers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, BatchNormalization, Flatten


class agente_IA():

    def __init__(self, historial_alimentos, factor_descuento, learning_rate, epsilon, N_acciones, N_memoria):
        self.historial_alimentos = historial_alimentos
        self.factor_descuento    = factor_descuento
        self.learning_rate       = learning_rate
        self.epsilon             = epsilon
        self.N_acciones          = N_acciones
        self.acciones_v          = np.arange(N_acciones) + 1
        self.N_memoria           = N_memoria
        self.N_minibatch         = N_memoria//10 if N_memoria > 10 else 2
        self.accuracy            = 0.5
        self.N_pred              = 0
        self.N_ok                = 0
        self.step                = 0

        # Inicialización de la memoria del agente sobre las últimas experiencias y del historial de recompensas percibidas
        self.memoria = []
        self.recompensa_v = []
        self.predicciones_Q = []


        

    def crear_modelo_Q(self, N_entradas, n_hidden_layers = [1024, 512, 256, 128, 64, 32, 16], activation = 'relu', dropout_factor = None, use_batch_norm = False):

        # Creación del modelo para predecir el valor de Q

        estado           = keras.layers.Input((N_entradas,), name='estado')
        filtro_acciones  = keras.layers.Input((self.N_acciones,), name='filtro_acciones')

        hidden_layers = []

        hidden_layers.append(keras.layers.Dense(n_hidden_layers[0], activation=activation)(estado))
        
        for n_layer in range(1,len(n_hidden_layers)):
            hidden_layers.append(keras.layers.Dense(n_hidden_layers[n_layer], activation=activation)(hidden_layers[-1]))

            if dropout_factor is not None:
                hidden_layers.append(Dropout(dropout_factor)(hidden_layers[-1]))

            if use_batch_norm:
                hidden_layers.append(BatchNormalization()(hidden_layers[-1])) 
                

        salida_Q   = keras.layers.Dense(self.N_acciones, activation = 'linear')(hidden_layers[-1])

        salida_modelo = keras.layers.multiply([salida_Q, filtro_acciones])


        modelo_Q = keras.models.Model([estado, filtro_acciones], salida_modelo)

        optimizador = keras.optimizers.Adam(lr=self.learning_rate, clipnorm=1.0)
        modelo_Q.compile(optimizador, loss='mean_squared_error')

        return modelo_Q



    def crear_dataset(self):
        estado_v, accion_v, recompensa_v, proximo_estado_v = [[],[],[],[]]

        input_estados  = []
        filtro_acciones_v = []
        target_v = []
        input_v = []

        idx_recuerdos_v = np.random.choice(np.arange(len(self.memoria)), self.N_minibatch)

        for i_recuerdo in idx_recuerdos_v:
            estado_v.append(self.memoria[i_recuerdo][0])
            accion_v.append(self.memoria[i_recuerdo][1])
            recompensa_v.append(self.memoria[i_recuerdo][2])
            proximo_estado_v.append(self.memoria[i_recuerdo][3])


        for j in range(len(estado_v)):

            filtro_acciones = np.zeros(self.N_acciones)
            aux_target = np.zeros(self.N_acciones)
            
##            filtro_acciones = np.ones(self.N_acciones)
##            aux_target = self.predecir(estado_v[j])
            
            
            filtro_acciones[accion_v[j]] = 1
            filtro_acciones_v.append(filtro_acciones)
            
            Q_v = self.predecir(proximo_estado_v[j])
            Q_max_proximo_estado = np.max(Q_v)


            estado_j = np.concatenate([estado_v[j][0]['valores'].reshape(-1), estado_v[j][1]['valores'].reshape(-1), [estado_v[j][2]], [estado_v[j][3]/(self.N_steps-1)]], axis=0)
            input_estados.append(estado_j)


            target = recompensa_v[j] + self.factor_descuento*Q_max_proximo_estado    ## Aplicando la ecuación de Ballman

            
            aux_target[accion_v[j]] = target

            target_v.append(aux_target)
        
        input_v = [input_estados, filtro_acciones_v]

        target_v = np.array(target_v)

        return input_v, target_v




    def entrenar(self, n_epochs=1, batch_size=64 , validation_split=0.2, verbose=0):

        input_v, target_v = self.crear_dataset()
        
        # Entrenamiento del modelo por n_epochs
        history_epoch = self.prox_modelo_Q.fit(input_v, target_v, epochs=n_epochs, verbose=verbose, validation_split=validation_split)

        return self.prox_modelo_Q, history_epoch




    def predecir(self, estado, accion = None):

        filtro_acciones = np.zeros(self.N_acciones)
##        filtro_acciones = np.ones(self.N_acciones)

        if accion is None:
            filtro_acciones += 1
        else:
            filtro_acciones[accion] = 1
            
        
        hist_v = estado[0]['valores']
        acciones_v = estado[1]['valores']
        accuracy = estado[2]
        step = estado[3]

        input_v = np.concatenate([hist_v.reshape(-1), acciones_v.reshape(-1), [accuracy], [step/(self.N_steps-1)]], axis=0)


        prediccion = self.modelo_Q.predict([[input_v], [filtro_acciones]])[0]

        self.predicciones_Q.append(prediccion.mean())

        return prediccion



        


        




