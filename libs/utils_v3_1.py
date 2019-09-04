#######################################################################################################
#####                       IMPORTACIÓN DE LIBRERÍAS Y RUTINAS A UTILIZAR                         #####
#######################################################################################################

import os, pickle
from sklearn.cluster import KMeans
import numpy as np



#######################################################################################################
#####                      UTILITARIOS PARA SALVAR/RECUPERAR ARCHIVOS                             #####
#######################################################################################################

def load_obj(file_d = './file.net', verbose=True):
    f = open(file_d, 'br')
    n = pickle.load(f)
    f.close()
    if verbose:
        print(' - Objeto', type(n), os.path.basename(file_d),'leído de disco.')
    return n

def dump_obj(n, file_d = './file.net', verbose=True):
    f = open(file_d, 'bw')
    pickle.dump(n, f)
    f.close()
    if verbose:
        print(' - Objeto', type(n), os.path.basename(file_d), 'salvado en disco.')



#######################################################################################################
#####                   FUNCIÓN PARA REALIZAR CLASIFICACIÓN UTILIZANDO K_MEANS                    #####
#######################################################################################################

def entrenar_kmeans(data_trn, data_tst, cluster_v, flag_print=False):

    diferencia_trn_tst_v = []
    std_v = []
    N_samples_trn_v= []
    N_samples_tst_v= []
    FM_trn_v= []
    FM_tst_v= []

    for n_clusters in cluster_v:
        # Creando el objeto del clasificador
        kmeans = KMeans(n_clusters=n_clusters)

        # Entrenando con el dataset de training
        kmeans = kmeans.fit(data_trn)
        
        # Probando con el dataset de trn
        prediction_trn = kmeans.predict(data_trn)

        # Probando con el dataset de test
        prediction_tst = kmeans.predict(data_tst)
        
        if flag_print:
            # Imprimiendo los centroides de los clusters luego del entrenamiento y los labels
            print('\nCentroides luego del entrenamiento:\n',  kmeans.cluster_centers_)
            print('\nLabels luego del entrenamiento:\n', kmeans.labels_)

            # Visualizando las clasificaciones luego del test
            print('\nClasificaciones de test:\n', prediction_tst)

        # Verificando la cantidad de Samples por grupo en entrenamiento y test
        N_samples_trn= np.zeros(n_clusters)
        N_samples_tst= np.zeros(n_clusters)
        
        error_trn= []
        error_tst= []

        dist_j = []
        
        for i in range(n_clusters):

            for j in range(n_clusters):
                if j != i:
                    dist_j.append(((kmeans.cluster_centers_[j] - kmeans.cluster_centers_[i])*(kmeans.cluster_centers_[j] - kmeans.cluster_centers_[i])).sum())
                
            aux_trn = kmeans.labels_ == i
            aux_tst = prediction_tst == i
            
            error_trn.append(np.power((data_trn[aux_trn] - kmeans.cluster_centers_[i])*(data_trn[aux_trn] - kmeans.cluster_centers_[i]), 0.5).sum())
            error_tst.append(np.power((data_tst[aux_tst] - kmeans.cluster_centers_[i])*(data_tst[aux_tst] - kmeans.cluster_centers_[i]), 0.5).sum())
            
            N_samples_trn[i] = aux_trn.sum()
            N_samples_tst[i] = aux_tst.sum()

        
        dist_clusters = np.mean(dist_j)
        
        N_samples_trn_v.append(N_samples_trn)
        N_samples_tst_v.append(N_samples_tst)

        prop_trn= N_samples_trn/N_samples_trn.sum()
        prop_tst= N_samples_tst/N_samples_tst.sum()
        
        FM_trn = (error_trn*prop_trn).mean()/dist_clusters
        FM_tst = (error_tst*prop_tst).mean()/dist_clusters

        diferencia_trn_tst = np.sum((prop_tst-prop_trn)*(prop_tst-prop_trn))
        diferencia_trn_tst_v.append(diferencia_trn_tst)
        
        std_v.append(np.std(N_samples_tst))
        FM_trn_v.append(FM_trn)
        FM_tst_v.append(FM_tst)

    N_samples_trn_v = np.array(N_samples_trn_v)
    N_samples_tst_v = np.array(N_samples_tst_v)
    FM_trn_v = np.array(FM_trn_v)/len(data_trn)
    FM_tst_v = np.array(FM_tst_v)/len(data_tst)

    
    return diferencia_trn_tst_v, std_v, N_samples_trn_v, N_samples_tst_v, FM_trn_v, FM_tst_v, kmeans
    
    

    

        
