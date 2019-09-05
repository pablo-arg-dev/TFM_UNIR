#######################################################################################################
#####                       IMPORTACIÓN DE LIBRERÍAS Y RUTINAS A UTILIZAR                         #####
#######################################################################################################

import numpy as np
import time
from matplotlib import pyplot as plt
import pandas as pd
import os, pickle, sys

from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist

from gensim import models

import warnings

sys.path.append(r'.\libs')
from entorno_v3_1   import *
from agente_IA_v3_2 import *
from utils_v3_1     import *



#######################################################################################################
#####                                COMIENZO DEL PROGRAMA PRINCIPAL                              #####
#######################################################################################################

if __name__ == "__main__":

    ############################################################
    # Inicialización de los parámetros utilizados en el código #
    ############################################################

    buscar_N_cluster = False               # Búsqueda de la cantidad mínima de clases para clasificar los alimentos en base a sus nutrientes
    cargar_mod_word_embedding = True       # Carga del modelo de word vectors utilizado
    prop_trn= 0.70                         # Proporcion de entrenamiento en el clasificador de alimentos base
    cluster_v = [8]                        # Cantidad de clases del clasificador

    verif_alim_marcas = False
    N_alim_min, N_alim_max = 5, 15         # Cantidad mínima y máxima de alimentos de marcas que una persona puede comer por día
    prop_min, prop_max     = 0.20, 0.80    # Proporción mínima y máxima de alimentos de marcas que una persona puede comer por día
    N_dias_hist_prueba  = 30*12            # Historial de alimentación de alimentos empaquetados para calcular la distancia que existe a la alimentación saludable (se propone 1 año para tener estadística suficiente)
    N_dias_hist_entorno = 7                # Cantidad de días utilizados para obtener el historial de alimentación de una persona que defina un entorno de preferencias en alimentos empaquetados

    N_episodes  = 200                      # Número de episodios a simular
    N_steps     = 100                      # Cantidad de pasos de tiempo por episodio
    N_memoria   = 2*N_steps                # Tamaño máximo del buffer de memoria de recuerdos
    prob_recomendacion_min = 0.1           # Límite inferior a la probabilidad de aceptar una recomendación por parte de las personas modeladas por el entorno
    prob_recomendacion_max = 0.9           # Límite superior a la probabilidad de aceptar una recomendación por parte de las personas modeladas por el entorno

    factor_descuento = 0.40                # Factor de descuento del agente
    learning_rate = 1E-3                   # Tasa de aprendizaje del agente
    train_step = N_steps                   # Tiempo de actualización de la red objetivo del esquema 'TD-target'


    exploracion_min, exploracion_max = 0.0, 1.0    # Límites mínimo y máximo del factor exploración/explotación

    decaimiento_exploracion =  N_episodes/5        # Constante del decaimiento exponencial del factor exploración/explotación para la estrategia del agente
                                                   # Se define tal que para step = N_steps el factor exploración/explotación sea menos del 1%


    N_alimentos_v        = [2]    # Tamaño de los conjuntos de alimentos a considerar en las pruebas
    
    do_graphs = True              # Generar gráficos para cada conjunto de alimentos
    print_alimentos = False       # Imprimir las opciones de alimentos y recomendaciones del agente en los últimos 2 pasos de tiempo por episodio




    
    ##########################################################################
    # Nombres de los archivos de las bases de datos utilizadas en el trabajo #
    ##########################################################################

    ds_filename = '../nut_data.dat'
    word_vectors_path = './Text_Sim/GoogleNews-vectors-negative300.bin'
    

    # Directorios de los archivos a leer
    dir_name_SR = '../SR-Leg_files/'
    dir_name_BFPD = '../BFPD-Leg_files/'

    # Nombres de los archivos a leer
    description_files_SR_v = ['FOOD_DES.txt', 'FOOTNOTE.txt', 'NUTR_DEF.txt']
    data_files_SR_v = ['DATA_SRC.txt', 'DATSRCLN.txt', 'DERIV_CD.txt', 'FD_GROUP.txt', 'LANGDESC.txt', 'LANGUAL.txt', 'NUT_DATA.txt', 'SRC_CD.txt', 'WEIGHT.txt']

    description_files_BFPD_v = ['Derivation_Code_Description.txt', 'Serving_Size.txt']
    data_files_BFPD_v        =['Nutrient.txt', 'Products.txt']




    ###############################
    # Carga de las bases de datos #
    ###############################

    # Carga del modelo de word vectors preentrenados en caso de solicitarlo
    if cargar_mod_word_embedding:
        if os.path.exists(word_vectors_path):
            print('\nCargando el modelo de word embedding')
            w = models.KeyedVectors.load_word2vec_format(word_vectors_path, binary=True)
            vocabulario = w.vocab
            word_embedding_data = [w,vocabulario]
            print('Modelo de word embedding cargado')
        else:
            print('ERROR! Falta el archivo de word vectors preentrenados', word_vectors_path)
            sys.exit()
    else:
        word_embedding_data = None



    # Carga de los archivos de descripcion de la base de datos de alimentos y nutrientes
    print('\nLeyendo la información de la base de datos de alimentos base y empaquetados..')
    description_SR   =  [np.array(pd.read_csv(dir_name_SR + file_name_i, header=None, encoding = 'ISO-8859-1', low_memory = False)) for file_name_i in description_files_SR_v]
    description_BFPD = [np.array(pd.read_csv(dir_name_BFPD + file_name_i, delimiter = ';', header=None, encoding = 'ISO-8859-1', low_memory = False)) for file_name_i in description_files_BFPD_v]
    data_BFPD        = [np.array(pd.read_csv(dir_name_BFPD + file_name_i, delimiter = ';', header=None, encoding = 'ISO-8859-1', low_memory = False)) for file_name_i in data_files_BFPD_v]
    print('\nArchivos de la base de datos cargados!')



    if not os.path.exists(ds_filename):
        
        data_SR =  [np.array(pd.read_csv(dir_name_SR + file_name_i, header=None, encoding = 'ISO-8859-1', low_memory = False)) for file_name_i in data_files_SR_v]


        print('\n\nGenerando dataset..')
        aux_food_idx = np.unique(data_SR[6][1:,0])                              ## Array con todos los alimentos posibles de dimensión (N_alim)
        aux_nut_idx = np.unique(data_SR[6][1:,1])                               ## Array con todos los nuetrientes posibles de dimensión (N_nut)
        nut_data = np.zeros([aux_food_idx.shape[0], aux_nut_idx.shape[0]])      ## Matriz de ceros de dimension (N_alim, N_nut) para cargar los nutrientes por tipo de alimento



        ## Carga de un array con los nombres de los nutrientes y otro con los números asignados a cada uno
        nut_names = []
        nut_nums  = []
        for nut_i in aux_nut_idx:
            nut_names.append(description_SR[2][1:,3][np.argmax(description_SR[2][1:,0] == nut_i)])
            nut_nums.append(description_SR[2][1:,0][np.argmax(description_SR[2][1:,0] == nut_i)])

        

        ## Carga de la matriz nut_data con la información de los nutrientes por tipo de alimento
        for i_food in range(data_SR[6][1:,0].shape[0]):
            nut_data[np.argwhere(data_SR[6][i_food+1,0] == aux_food_idx)[0,0], np.argwhere(data_SR[6][i_food+1,1] == aux_nut_idx)[0,0]] = data_SR[6][i_food+1,2]



        ## Guardado de la matriz nut_data y la lista nut_names en disco
        dump_obj([nut_data,nut_names, nut_nums, aux_nut_idx], ds_filename, verbose=False)
        print('\nArchivo del dataset guardado en disco')

    else:
        ## Recuperación de la matriz nut_data y la lista nut_names desde disco
        nut_data, nut_names, nut_nums, aux_nut_idx = load_obj(ds_filename, verbose=False)
        print('\nArchivo del dataset cargado de disco')




    ## Borrado de las características que no se encuentran en el target nutricional    
    nut_to_del_v = []
    target_nut = []
    for key in nut_target.keys():
        if nut_target[key] is None:
            nut_to_del = description_SR[2][1:,0][np.argmax(description_SR[2][1:,3] == key)]
            nut_to_del_v.append(np.argmax(aux_nut_idx == nut_to_del))
        else:
            target_nut.append(nut_target[key])

    
    nut_data    = np.delete(nut_data, nut_to_del_v, axis=1)
    nut_names   = np.delete(nut_names, nut_to_del_v, axis=0)
    nut_nums    = np.delete(nut_nums, nut_to_del_v, axis=0)
    aux_nut_idx = np.delete(aux_nut_idx, nut_to_del_v, axis=0)
    target_nut  = np.array(target_nut)



    # Seleccionamos los conjuntos de training y test de los alimentos base y normalizamos los datos
    n_trn= round(len(nut_data)*prop_trn)
    n_tst= round(len(nut_data)*(1-prop_trn))
    print('N_samples_trn:',n_trn, '\tN_samples_tst:', n_tst)

    trn_random_idx = np.random.choice(len(nut_data), n_trn, replace=False)  # En trn buscamos n_trn samples de forma aleatoria para filtrar features y target
    tst_random_idx =  np.delete(np.arange(len(nut_data)), trn_random_idx)   # Utilizamos para tst los samples que no utilizamos para trn



    aux_features_max_trn = np.max(nut_data[trn_random_idx], axis= 0)
    aux_features_min_trn = np.min(nut_data[trn_random_idx], axis= 0)
    features_norm_trn = (nut_data[trn_random_idx] - aux_features_min_trn)/(aux_features_max_trn - aux_features_min_trn)
    
    aux_features_max_tst = np.max(nut_data[tst_random_idx], axis= 0)
    aux_features_min_tst = np.min(nut_data[tst_random_idx], axis= 0)
    features_norm_tst = (nut_data[tst_random_idx] - aux_features_min_tst)/(aux_features_max_tst - aux_features_min_tst)





    #################################################################################
    # Comienzo del trabajo de clasificación de alimentos / aprendizaje por refuerzo #
    #################################################################################

    if buscar_N_cluster:
        
        # Probando con distintos tamaños de clusters hasta 2.0*sqrt(N_samples) y graficando..
        cluster_v = np.arange(2,int(3.0*np.sqrt(nut_data.shape[1])))
        info_trn_tst = entrenar_kmeans(features_norm_trn, features_norm_tst, cluster_v, flag_print=False)
        diferencia_trn_tst_v, std_v, N_samples_trn_v, N_samples_tst_v, FM_trn_v, FM_tst_v, kmeans= info_trn_tst
        plt.figure(1)
        plt.title('Figura de mérito para la clasificación de alimentos base en training vs. N_clusters')
        A= plt.plot(cluster_v, FM_trn_v)
        plt.figure(2)
        plt.title('Figura de mérito para la clasificación de alimentos base en test vs. N_clusters')
        B= plt.plot(cluster_v, FM_tst_v)
        plt.show()


    else:

        #####################################################################################################
        ##   COMIENZO DEL ALGORITMO DE IA PARA CLASIFICAR ALIMENTOS EN BASE A SU INFORMACIÓN NUTRICIONAL   ##
        #####################################################################################################
        
        # Asumiendo un numero de clusters para clasificar a los alimentos por su aporte nutricional..
        print('\nClasificando los alimentos base en', cluster_v[0], 'categorías')
        info_trn_tst = entrenar_kmeans(features_norm_trn, features_norm_tst, cluster_v, flag_print=False)
        diferencia_trn_tst_v, std_v, N_samples_trn_v, N_samples_tst_v, FM_trn_v, FM_tst_v, kmeans= info_trn_tst

        coordenadas_clusters = kmeans.cluster_centers_

        print('\nSamples por clase en entrenamiento:', N_samples_trn_v[-1])
        porcentajes_clases_trn = np.round(N_samples_trn_v[-1]/N_samples_trn_v[-1].sum()*100,1)
        print('Proporcion de samples por clase en entrenamiento (%):', porcentajes_clases_trn)
        
        print('\nSamples por grupo en test:', N_samples_tst_v[-1])
        porcentajes_clases_tst = np.round(N_samples_tst_v[-1]/N_samples_tst_v[-1].sum()*100, 1)
        print('Proporcion de samples por clase en test (%):', porcentajes_clases_tst)




        # Construccion del target nutricional a partir de las coordenadas de los centros de los clusters
        target_nut_norm = (target_nut/masa_inicial_comidas_g*100 - aux_features_min_tst)/(aux_features_max_tst - aux_features_min_tst)   # Debe normalizarse target_nut para considerar los nutrientes o features en 100 gramos.
        clase_target_inicial = kmeans.predict(np.expand_dims(target_nut_norm, axis=0))      

        coordenadas_target_inicial = cdist(np.expand_dims(target_nut_norm, axis=0), coordenadas_clusters)[0]
        print('\nClase_target_inicial:', clase_target_inicial[0], '\nCoordenadas_target_inicial:', coordenadas_target_inicial)


        pocentaje_ok_tst = porcentajes_clases_tst[clase_target_inicial[0]]
        print('\nPorcenjate de alimentos base dentro de la clase del target nutricional:', pocentaje_ok_tst, '%')
        


        # True para definir la proporción de días en que se logra la alimentación saludable dentro de la clase del target nutricional recomendado
        if verif_alim_marcas:
            feature_agglom = None
      
            print('\n\n\nCalculando la proporción de días en que se logra la ingesta recomendada con alimentos de marcas..')
            print('\nSe propone un historial de comidas random utilizando alimentos de marcas')
            print('\nConstruyendo el historial de alimentación durante', N_dias_hist_prueba, 'días de prueba')

            entorno_agente = entorno(None, nut_nums, None, aux_features_max_trn, aux_features_min_trn,
                                     None, None, None)


            # Construccion de un historial ficticio utilizando alimentos de marcas
            hist_v = entorno_agente.construir_historial_alim(N_dias_hist_prueba, verif_alim_marcas, N_alim_max, N_alim_min, prop_max, prop_min, data_BFPD)

            # Predicción de las clases del historial ficticio con alimentos de marcas
            clases_historial_alim = kmeans.predict(hist_v['valores'])
            porcentaje_hist_clase_target = np.round(100*(clases_historial_alim == clase_target_inicial[0]).mean(), 2)
            print('\n\nPorcentaje de días con alimentación dentro de la clase del target nutricional utilizando alimentos de marcas:', porcentaje_hist_clase_target, '%')



        else:

            #############################################################################################
            ##   COMIENZO DEL ALGORITMO DE IA PARA ELEGIR ALIMENTOS EN BASE A LA INGESTA RECOMENDADA   ##
            #############################################################################################

            feature_agglom = FeatureAgglomeration(n_clusters=cluster_v[-1])

            feature_agglom.fit(nut_data)

            features_reduced = feature_agglom.transform(nut_data)

            aux_features_max = np.max(features_reduced, axis=0)
            aux_features_min = np.min(features_reduced, axis=0)

            target_nut_norm = ((feature_agglom.transform(np.expand_dims(target_nut/masa_inicial_comidas_g*100, axis=0)) - aux_features_min)/(aux_features_max-aux_features_min))[0]



            pendiente_aprendizaje_v   = []
            recompensa_m = []
            
            for N_alimentos in N_alimentos_v:


                # Creación de la carpeta para salvar las Figuras y datos de cada conjunto de alimentos por separado
                dump_folder = './Figuras_' + str(int(time.time())) + '_' + str(N_alimentos) + ' alimentos' + '/'

                print('Creando directorio:', dump_folder)
                os.mkdir(dump_folder)
    


                recompensas_acumuladas_v  = []
                confianza_entorno_final_v = []

                recompensa_m.append([])
                pendiente_aprendizaje_v.append([])
                

                # Cadena para identificar los archivos de cada corrida del programa en función de sus parámetros
                dump_name = '_' + str(N_episodes) + ' Ep_' + str(N_steps) + ' Steps_' + str(N_alimentos) + ' Alim_' + str(N_memoria) + ' Mem_' + str(prob_recomendacion_min) + ' ProbRec_' + str(factor_descuento) + ' FD_' + str(learning_rate) + ' LR' + '.dat'

    
                

                # Definición del objeto que representa un ser humano ficticio que constituye el entorno
                # Se define en base a su historial de preferencias y a la cantidad de opciones que tiene para elegir un alimento en un instante
                entorno_agente = entorno(historial_alimentos = None,  nut_nums=nut_nums, feature_agglom=feature_agglom, aux_features_max=aux_features_max, aux_features_min=aux_features_min,
                                         target_nut = target_nut_norm,  N_acciones = N_alimentos, info_alimentos = data_BFPD, prob_recomendacion = prob_recomendacion_min, similarity_type = 'word_embedding', word_embedding_data = word_embedding_data)
                
                entorno_agente.N_steps = N_steps


                # Definición del objeto que representa al agente de IA. Lo define el historial de alimentos del humano, su factor de descuento y tasa de aprendizaje
                agente = agente_IA(historial_alimentos = None, factor_descuento = factor_descuento, learning_rate = learning_rate, epsilon = exploracion_max, N_acciones = N_alimentos, N_memoria = N_memoria)

                # Creación e inicialización del modelo para predecir los valores Q(estado,accion)
                agente.modelo_Q = agente.crear_modelo_Q(N_entradas = (N_dias_hist_entorno + N_alimentos)*target_nut_norm.shape[-1] + 2, activation = 'relu',  dropout_factor = None, use_batch_norm = False)    # arg --> (N_entradas, n_hidden_layers = [128,64], activation = 'relu', dropout_factor = None)

                agente.prox_modelo_Q = agente.modelo_Q



                # Construccion de un historial ficticio utilizando alimentos de marcas
                hist_v_i = entorno_agente.construir_historial_alim(N_dias_hist_entorno, verif_alim_marcas, N_alim_max, N_alim_min, prop_max, prop_min, data_BFPD)

                agente.historial_alimentos         = hist_v_i
                entorno_agente.historial_alimentos = hist_v_i

                agente.N_steps = N_steps




                ### Comienzo del bucle que barre los distintos episodios para el agente

                for episode_i in range(1, N_episodes + 1):

                    print('\nEpisodio:', episode_i)



                    # Construccion de un historial ficticio utilizando alimentos de marcas
                    hist_v_i = entorno_agente.construir_historial_alim(N_dias_hist_entorno, verif_alim_marcas, N_alim_max, N_alim_min, prop_max, prop_min, data_BFPD)

                    agente.historial_alimentos         = hist_v_i
                    entorno_agente.historial_alimentos = hist_v_i

                    # Borrado de la codificación del historial de alimentos anterior como word embedding
                    entorno_agente.hist_word_vectors = None


                    # Definición de la evolución temporal del factor exploración/explotación como exponencial respecto al número de episodio
                    agente.epsilon = exploracion_min + (exploracion_max-exploracion_min)*np.exp(-(episode_i-1)/decaimiento_exploracion)

                    entorno_agente.prob_recomendacion = prob_recomendacion_min
                    agente.accuracy = 0.5
                    agente.N_pred = 0
                    agente.N_ok = 0


                    entorno_agente.confianza_v = []
                    entorno_agente.hist_confianza_v.append([])
                    agente.recompensa_v.append([])

                    # Se eligen N_alimentos de marcas de forma random sobre los cuales el agente debe expedir su decisión
                    alimentos_seleccionados_v = entorno_agente.elegir_alimentos(use_random = True)


                    if len(entorno_agente.confianza_v) == 0:
                        hist_v = hist_v_i

                    # El estado en cada epoch para el agente se construye por el historial de alimentos y aquellos sobre los cuales debe elegir
                    estado = [hist_v, alimentos_seleccionados_v, 0.5, 1]




                    ### Comienzo del bucle que los diferentes pasos dentro de cada episodio

                    for step in range(1, N_steps + 1):

                        if step == 1:
                            print('\n\nPorcentaje exploración:', round(agente.epsilon*100, 2), '%')

                        if not print_alimentos:
                            print('\nEp.:', episode_i, '\tStep:', step)

                        agente.step = step


                        # Definición de un número aleatorio para explorar o no el dominio mediante la selección de una acción random
                        random_num = np.random.random()

                        if random_num < agente.epsilon:
                            accion_agente = np.random.choice(np.arange(N_alimentos), 1)[0]
                        else:
                            accion_agente = np.argmax(agente.predecir(estado))

                            
                        

                        recompensa, accion_humano, rec_v= entorno_agente.calc_recompensa(estado, accion_agente)


                        if print_alimentos and step >= N_steps-1:
                            print('\nOpciones de alimentos Step', step)
                            for i_alim in range(len(alimentos_seleccionados_v['nombres'])): print('Opción', i_alim+1, ':', alimentos_seleccionados_v['nombres'][i_alim])
                            print('\nRecomendación agente: Opción', accion_agente+1)
                            print('Selección de la persona del entorno: Opción', accion_humano+1)
                            print('Recompensa percibida en este paso de tiempo:', recompensa)


                        agente.recompensa_v[-1].append(recompensa)


                        entorno_agente.hist_confianza_v[-1].append(entorno_agente.prob_recomendacion)


                        acc_v = np.ones(agente.N_acciones)*(agente.N_ok)/(agente.N_pred+1)
                        

                        # Actualización del nivel de confianza del entorno y del agente
                        if accion_agente == accion_humano:
                            entorno_agente.confianza_v.append(1)
                            agente.N_pred += 1
                            agente.N_ok += 1
                        else:
                            entorno_agente.confianza_v.append(-1)
                            agente.N_pred += 1

                        promedio_confianza_v = np.mean(entorno_agente.confianza_v)


                        
                        agente.accuracy = agente.N_ok/agente.N_pred

                        acc_v[accion_agente] = agente.accuracy



                        if len(entorno_agente.confianza_v) > 0:
                            entorno_agente.prob_recomendacion = np.mean(entorno_agente.confianza_v)
                            


                        ## Establecimiento de los límites inferior y superior para la prob. de aceptar cada recomendación por las personas del entorno
                        if entorno_agente.prob_recomendacion < prob_recomendacion_min:
                            entorno_agente.prob_recomendacion = prob_recomendacion_min
                        if entorno_agente.prob_recomendacion > prob_recomendacion_max:
                            entorno_agente.prob_recomendacion = prob_recomendacion_max

                        

                        # Se eligen N_alimentos de marcas de forma random sobre los cuales el agente debe expedir su decisión
                        alimentos_seleccionados_v = entorno_agente.elegir_alimentos(use_random = True)
                        
                        proximo_estado_v = [[hist_v, alimentos_seleccionados_v, acc, step] for acc in acc_v]

                        proximo_estado = proximo_estado_v[accion_agente]

                        agente.memoria.append([estado, accion_agente, rec_v[accion_agente], proximo_estado_v[accion_agente]])



                        # Si la memoria llega a la longitud máxima permitida, el agente se olvida de la experiencia mas vieja
                        if len(agente.memoria) > agente.N_memoria:
                            agente.memoria.pop(0)
                            


                        estado = proximo_estado


                        if len(agente.memoria) >= agente.N_minibatch:
                            agente.prox_modelo_Q, history_step = agente.entrenar(verbose=0, n_epochs=1)



                        if step % train_step == 0:
                            agente.modelo_Q = agente.prox_modelo_Q

                            
                
                entorno_agente.hist_confianza_v = np.array(entorno_agente.hist_confianza_v)



                recompensa_v = np.cumsum(agente.recompensa_v, axis=-1)
                recompensas_acumuladas = recompensa_v[:,-1]
                recompensas_acumuladas_v.append(recompensas_acumuladas)

                confianza_entorno_final_v.append(entorno_agente.hist_confianza_v[-1])
                recompensa_m[-1].append(recompensa_v[-1])

                
                
                ### Graficación de los resultados obtenidos y salvaguarda de la información
                if do_graphs:

                    ## Confianza inicial - final del entorno (promediando en 5 episodios)
                    plt.figure(1)
                    plt.title('Confianza entorno inicial vs. final')

                    hist_conf_inicial = np.array(entorno_agente.hist_confianza_v[:5]).mean(axis=0)
                    plt.plot(hist_conf_inicial, 'r', label='Primeros 5 episodios')
                    
                    hist_conf_final = np.array(entorno_agente.hist_confianza_v[-5:]).mean(axis=0)
                    plt.plot(hist_conf_final, 'g', label='Últimos 5 episodios')
                    
                    plt.xlabel('Step')
                    plt.ylabel('Confianza del entorno')
                    plt.legend(loc=0, fontsize = 'small')
                    plt.savefig(dump_folder + 'hist_conf_i_f' + dump_name + '.png', bbox_inches='tight')

                    dump_obj(hist_conf_inicial, dump_folder + 'hist_conf_inicial' + dump_name)
                    dump_obj(hist_conf_final, dump_folder + 'hist_conf_final' + dump_name)



                    
                    ## Predicciones de Q
                    plt.figure(2)
                    plt.title('Predicciones Q en training')
                    plt.plot(agente.predicciones_Q)

                    plt.xlabel('N training')
                    plt.ylabel('Q')

                    plt.savefig(dump_folder + 'predicciones_Q' + dump_name + '.png', bbox_inches='tight')
                    
                    dump_obj(agente.predicciones_Q, dump_folder + 'predicciones_Q' + dump_name)



                    ## Recompensas acumuladas
                    plt.figure(3)
                    plt.title('Recompensas acumuladas vs. episodio')
                    recompensa_v = np.cumsum(agente.recompensa_v, axis=-1)
                    recompensas_acumuladas = recompensa_v[:,-1]
                    plt.plot(recompensas_acumuladas)

                    plt.xlabel('Episodio')
                    plt.ylabel('Recompensas acumuladas')

                    plt.savefig(dump_folder + 'recompensas_acumuladas' + dump_name + '.png', bbox_inches='tight')
                    
                    dump_obj(recompensas_acumuladas, dump_folder + 'recompensas_acumuladas' + dump_name)




                    plt.figure(4)
                    plt.title('Recompensas inicial y final vs. Step')

                    plt.plot(recompensa_v[0], 'r', label='Episodio 1')
                    plt.plot(recompensa_v[-1], 'g', label='Episodio ' + str(N_episodes))

                    plt.xlabel('Step')
                    plt.ylabel('Recompensa acumulada')
                    plt.legend(loc=0, fontsize = 'small')

                    plt.savefig(dump_folder + 'recompensas_inicial_final' + dump_name + '.png', bbox_inches='tight')

                    dump_obj(recompensa_v[0], dump_folder + 'recompensa_inicial' + dump_name)
                    dump_obj(recompensa_v[-1], dump_folder + 'recompensa_final' + dump_name)




                    ## Recompensas inicial - final acumuladas por el agente (promediando en 5 episodios)
                    plt.figure(5)
                    plt.title('Recompensas acumuladas al inicio y al final de las simulaciones')

                    prom_i = recompensa_v[:5].mean(axis=0)
                    prom_f = recompensa_v[-5:].mean(axis=0)
                    
                    plt.plot(prom_i, 'r', label='Primeros 5 episodios')
                    plt.plot(prom_f, 'g', label='Últimos 5 episodios')
                    
                    plt.xlabel('Step')
                    plt.ylabel('Promedio de recompensas acumuladas')
                    plt.legend(loc=0, fontsize = 'small')
                    plt.savefig(dump_folder + 'promedio_recompensas_i_f' + dump_name + '.png', bbox_inches='tight')

                    dump_obj(prom_i, dump_folder + 'prom_recompensas_inicial' + dump_name)
                    dump_obj(prom_f, dump_folder + 'prom_recompensas_final' + dump_name)



                    
                    plt.figure(6)
                    plt.title('Confianza del entorno vs. Step')

                    plt.plot(entorno_agente.hist_confianza_v[0], 'r', label='Episodio 1')
                    plt.plot(entorno_agente.hist_confianza_v[-1], 'g', label='Episodio ' + str(N_episodes))

                    plt.xlabel('Step')
                    plt.ylabel('Confianza')
                    plt.legend(loc=0, fontsize = 'small')

                    plt.savefig(dump_folder + 'conf_i_f' + dump_name + '.png', bbox_inches='tight')

                    dump_obj(entorno_agente.hist_confianza_v[0], dump_folder + 'conf_inicial' + dump_name)
                    dump_obj(entorno_agente.hist_confianza_v[-1], dump_folder + 'conf_final' + dump_name)


                    ## Se eliminan todos los gráficos del buffer
                    plt.close('all')

                ## Se muestran los gráficos realizados
    ##            plt.show()

            
    sys.exit() 

#######################################################################################################
#####                                FIN DEL PROGRAMA PRINCIPAL                                   #####
#######################################################################################################
