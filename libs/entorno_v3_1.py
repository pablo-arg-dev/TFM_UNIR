#######################################################################################################
#####                       IMPORTACIÓN DE LIBRERÍAS Y RUTINASA UTILIZAR                          #####
#######################################################################################################

import numpy as np
import os, sys
from scipy.spatial.distance import cdist
from gensim import models

from pytrends.request import TrendReq
pytrend = TrendReq()

import warnings




########################################################################################################
#####         DICCIONARIO DE PAISES PARA REALIZAR BÚSQUEDAS DE TENDENCIA EN GOOGLE TRENDS          #####
#####                                               -                                              #####
#####  TARGET NUTRICIONAL DEL ENTORNO CON LA INFORMACIÓN DE LAS VITAMINAS Y MINERALES DE LA DIETA  #####
#####                                               -                                              #####
#####               CANTIDAD DE COMIDA DIARIA EN GRAMOS TOMADA COMO REFERENCIA                     #####
########################################################################################################

geo_d = {'Alemania':'DE', 'Arabia Saudita':'SA', 'Argentina':'AR', 'Australia':'AU', 'Belgica':'BE', 'Brasil':'BR', 'Canada':'CA', 'Catar':'QA', 'China':'CN', 'Corea del Sur':'KR', 'Emiratos Arabes':'AE', 'España':'ES', 'Estados Unidos':'US', 'Francia':'FR', 'Grecia':'GR', 'Hong Kong':'HK', 'India':'IN', 'Israel':'IL', 'Italia':'IT', 'Japon':'JP', 'Malasia':'MY', 'Mexico':'MX', 'Noruega':'NO', 'Nueva Zelanda':'NZ', 'Holanda':'NL', 'Pakistan':'PK', 'Reino Unido': 'GB', 'Rusia':'RU', 'Suecia':'SE', 'Suiza':'CH', 'Taiwan':'TW'}

nut_target = {'Protein':56, 'Total lipid (fat)':30, 'Carbohydrate, by difference':130, 'Ash':None, 'Energy':2200,
              'Starch':45, 'Sucrose':None, 'Glucose (dextrose)':None, 'Fructose':None, 'Lactose':None,
              'Maltose':None, 'Alcohol, ethyl':None, 'Water':None, 'Caffeine':None, 'Theobromine':None, 'Energy_KJ':None,
              'Sugars, total':None, 'Galactose':None, 'Fiber, total dietary':30.8, 'Calcium, Ca':1000, 'Iron, Fe':8,
              'Magnesium, Mg':420, 'Phosphorus, P':700, 'Potassium, K':4700, 'Sodium, Na':2300, 'Zinc, Zn':11, 'Copper, Cu':0.9,
              'Fluoride, F':None, 'Manganese, Mn':2.3, 'Selenium, Se':55, 'Vitamin A, IU':None, 'Retinol':None, 'Vitamin A, RAE':900,
              'Carotene, beta':None, 'Carotene, alpha':None, 'Vitamin E (alpha-tocopherol)':15, 'Vitamin D':600, 'Vitamin D2 (ergocalciferol)':None,
              'Vitamin D3 (cholecalciferol)':None, 'Vitamin D (D2 + D3)':None, 'Cryptoxanthin, beta':None, 'Lycopene':None, 'Lutein + zeaxanthin':None,
              'Tocopherol, beta':None, 'Tocopherol, gamma':None, 'Tocopherol, delta':None, 'Tocotrienol, alpha':None, 'Tocotrienol, beta':None,
              'Tocotrienol, gamma':None, 'Tocotrienol, delta':None, 'Vitamin C, total ascorbic acid':90,
              'Thiamin':1.2, 'Riboflavin':1.3, 'Niacin':16, 'Pantothenic acid':10, 'Vitamin B-6':1.3, 'Folate, total':None,
              'Vitamin B-12':2.4, 'Choline, total':550, 'Menaquinone-4':None, 'Dihydrophylloquinone':None, 'Vitamin K (phylloquinone)':120,
              'Folic acid':400, 'Folate, food':None, 'Folate, DFE':400, 'Betaine':None, 'Tryptophan':None, 'Threonine':None, 'Isoleucine':None,
              'Leucine':None, 'Lysine':None, 'Methionine':None, 'Cystine':None, 'Phenylalanine':None, 'Tyrosine':None, 'Valine':None, 'Arginine':None,
              'Histidine':None, 'Alanine':None, 'Aspartic acid':None, 'Glutamic acid':None, 'Glycine':None, 'Proline':None, 'Serine':None,
              'Hydroxyproline':None, 'Vitamin E, added':None, 'Vitamin B-12, added':None, 'Cholesterol':None, 'Fatty acids, total trans':None,
              'Fatty acids, total saturated':20, '4:0':None, '6:0':None, '8:0':None, '10:0':None, '12:0':None, '14:0':None, '16:0':None,
              '18:0':None, '20:0':None, '18:1 undifferentiated':None, '18:2 undifferentiated':None, '18:3 undifferentiated':None,
              '20:4 undifferentiated':None, '22:6 n-3 (DHA)':None, '22:0':None, '14:1':None, '16:1 undifferentiated':None, '18:4':None, '20:1':None,
              '20:5 n-3 (EPA)':None, '22:1 undifferentiated':None, '22:5 n-3 (DPA)':None, 'Phytosterols':None, 'Stigmasterol':None,
              'Campesterol':None, 'Beta-sitosterol':None, 'Fatty acids, total monounsaturated':10, 'Fatty acids, total polyunsaturated':10,
              '15:0':None, '17:0':None, '24:0':None, '16:1 t':None, '18:1 t':None, '22:1 t':None, '18:2 t not further defined':None,
              '18:2 i':None, '18:2 t,t':None, '18:2 CLAs':None, '24:1 c':None, '20:2 n-6 c,c':None, '16:1 c':None, '18:1 c':None,
              '18:2 n-6 c,c':None, '22:1 c':None, '18:3 n-6 c,c,c':None, '17:1':None, '20:3 undifferentiated':None,
              'Fatty acids, total trans-monoenoic':None, 'Fatty acids, total trans-polyenoic':None, '13:0':None, '15:1':None,
              '18:3 n-3 c,c,c (ALA)':None, '20:3 n-3':None, '20:3 n-6':None, '20:4 n-6':None, '18:3i':None, '21:5':None, '22:4':None,
              '18:1-11 t (18:1t n-7)':None}

masa_inicial_comidas_g = 1200






#######################################################################################################
##### DEFINICION DEL ENTORNO - MODELADO DE LA RESPUESTA DEL HUMANO EN FUNCIÓN DE SUS PREFERENCIAS #####
##### Y DE LA PROBABILIDAD QUE TIENE DE ACEPTAR UNA RECOMENDACIÓN                                 #####
##### (LA PROB. ESTÁ BASADA EN LA CONFIANZA QUE LE TENGA AL ALGORITMO EN FUNCION DE SUS ACIERTOS) #####
#######################################################################################################

class entorno():
    

    def __init__(self, historial_alimentos, nut_nums, feature_agglom, aux_features_max, aux_features_min, target_nut, N_acciones, info_alimentos, prob_recomendacion = 0.1, valoracion_temp = 0.1, similarity_type = None, word_embedding_data = None, lugar = 'Estados Unidos'):

        # Definicion de las propiedades del humano, que representa oarte del entorno del problema
        self.historial_alimentos = historial_alimentos       # Historial de alimentos que define sus preferencias en los últimos X días
        self.N_acciones          = N_acciones                # Cantidad de alimentos sobre los que puede elegir en cada oportunidad
        self.prob_recomendacion  = prob_recomendacion              # Probabilidad inicial de aceptar una recomendación sobre un alimento
        self.valoracion_temp     = valoracion_temp           # Preferencia temporal por la comida (dando preferencias a alimentos no ingeridos en mucho tiempo)
        self.lugar               = lugar                     # Lugar desde donde se buscarían las tendencias de google en caso de habilitar esta opción
        self.nut_nums            = nut_nums
        self.feature_agglom      = feature_agglom
        self.aux_features_max    = aux_features_max
        self.aux_features_min    = aux_features_min
        self.target_nut          = target_nut
        self.info_alimentos      = info_alimentos
        self.confianza_v         = []
        self.hist_confianza_v    = []
        self.similarity_type     = similarity_type
        self.hist_word_vectors   = None

        if similarity_type == 'word_embedding' and word_embedding_data is None:
            print('\nError! Falta la información de word_embedding_data.')
            sys.exit()

        if word_embedding_data is not None:
            self.w               = word_embedding_data[0]
            self.vocabulario     = word_embedding_data[1]


        

    def construir_historial_alim(self, N_dias_hist, verif_alim_marcas, N_alim_max, N_alim_min, prop_max, prop_min, data_BFPD):

        
        hist_v = {'nombres':[], 'valores':[]}
        for dia_i in range(N_dias_hist):
            if verif_alim_marcas:
                print('Día:', dia_i+1)

            hist_v['nombres'].append([])
            N_alim_dia_i = N_alim_min + int((N_alim_max-N_alim_min+1)*np.random.random())    # Cantidad de productos que una persona comió por día

            aux_prop_alim_dia_i = np.array([prop_min + (prop_max-prop_min)*np.random.random() for k in range(N_alim_dia_i)])    # Cálculo de las proporciones random ingeridas por día
            prop_alim_dia_i = aux_prop_alim_dia_i/aux_prop_alim_dia_i.sum()


            idx_alim_dia_i = np.random.choice(data_BFPD[1].shape[0]-1, N_alim_dia_i, replace = False)   # Índices de los alimentos ingeridos por día

            nut_dia_i = np.zeros(self.nut_nums.shape[0])

            for j_idx in range(len(idx_alim_dia_i)):
                NDB_alim_j = data_BFPD[1][1:,0][idx_alim_dia_i[j_idx]]              # Número NDB del alimento j
                long_name_alim_j = data_BFPD[1][1:,1][idx_alim_dia_i[j_idx]]        # long_name del alimento j
                aux_idx_nut_j = data_BFPD[0][1:,0] == NDB_alim_j                    # Índices de los nutrientes del alimento j
                nut_data_alim_j = data_BFPD[0][1:][aux_idx_nut_j]                   # Nombres de los nutrientes del alimento j
                
                for k_nut in range(len(nut_data_alim_j)):
                    if nut_data_alim_j[k_nut,1] in self.nut_nums:
                        nut_dia_i[np.argmax(self.nut_nums == nut_data_alim_j[k_nut,1])] += float(nut_data_alim_j[k_nut][4])*prop_alim_dia_i[j_idx]

                hist_v['nombres'][-1].append(long_name_alim_j)

            hist_v['valores'].append(nut_dia_i) 

        if self.feature_agglom is not None:
            hist_v['valores'] = (self.feature_agglom.transform(np.array(hist_v['valores'])) - self.aux_features_min)/(self.aux_features_max-self.aux_features_min)

            hist_v['valores'][hist_v['valores'] > 1] = 1.00
            hist_v['valores'][hist_v['valores'] < 0] = 0.00
        else:
            hist_v['valores'] = (np.array(hist_v['valores']) - self.aux_features_min)/(self.aux_features_max-self.aux_features_min)
        
        return hist_v





    def elegir_alimentos(self, use_random = True):

        alimentos_seleccionados_v = {'nombres':[], 'valores':[]}

        if use_random:
            idx_alim_v = np.random.choice(self.info_alimentos[1].shape[0]-1, self.N_acciones, replace = False)
        
        for j_idx in range(len(idx_alim_v)):
            nut_alim_j = np.zeros(self.nut_nums.shape[0])
            NDB_alim_j = self.info_alimentos[1][1:,0][idx_alim_v[j_idx]]
            long_name_alim_j = self.info_alimentos[1][1:,1][idx_alim_v[j_idx]]
            aux_idx_nut_j = self.info_alimentos[0][1:,0] == NDB_alim_j
            nut_data_alim_j = self.info_alimentos[0][1:][aux_idx_nut_j]

            for k_nut in range(len(nut_data_alim_j)):
                if nut_data_alim_j[k_nut,1] in self.nut_nums:
                    nut_alim_j[np.argmax(self.nut_nums == nut_data_alim_j[k_nut,1])] += float(nut_data_alim_j[k_nut][4])


            alimentos_seleccionados_v['nombres'].append(long_name_alim_j)
            alimentos_seleccionados_v['valores'].append(nut_alim_j)


        alimentos_seleccionados_v['valores'] = (self.feature_agglom.transform(np.array(alimentos_seleccionados_v['valores'])) - self.aux_features_min)/(self.aux_features_max-self.aux_features_min)

        alimentos_seleccionados_v['valores'][alimentos_seleccionados_v['valores'] > 1] = 1.00
        alimentos_seleccionados_v['valores'][alimentos_seleccionados_v['valores'] < 0] = 0.00

        return alimentos_seleccionados_v





    def wordEmb_sim(self, posible_accion, historial_v, idioma = 'inglés'):

        palabras_comunes_v = ['of', 'the', 'a', 'an', 'to', 'in']

        if idioma != 'inglés':
            print('\nError! "idioma":', idioma, 'no soportado por el momento')
            sys.exit()


        # Codifica de una posible acción en los word vectors incluidos en la base de datos
        accion_words = posible_accion.split()
        accion_word_vectors = []
        
        for word_i in accion_words:
            if word_i in self.vocabulario and word_i not in palabras_comunes_v:
                accion_word_vectors.append(self.w.get_vector(word_i))

                
        # Se codifica el historial en word vectors una sola vez por episodio
        if self.hist_word_vectors is None:
            
            hist_word_vectors = []
            self.N_alim_v = []
            
            for i in range(len(historial_v)):
                
                for j in range(len(historial_v[i])):

                    alim_word_vectors = []
                    aux_nombre_alimento = historial_v[i][j].split()

                    for word_i in aux_nombre_alimento:
                        if word_i in self.vocabulario and word_i not in palabras_comunes_v:
                            alim_word_vectors.append(self.w.get_vector(word_i))
                            
                    
                    hist_word_vectors.append(alim_word_vectors)
                self.N_alim_v.append(len(historial_v[i]))

            self.hist_word_vectors = hist_word_vectors

        else:
            hist_word_vectors = self.hist_word_vectors


        # Inicialización en cero de la matriz de similaridad
        similarity_matrix = np.zeros([len(accion_word_vectors), len(hist_word_vectors)])


        # Cálculo de los valores de la matriz de similaridad
        for i_word_vector in range(len(accion_word_vectors)):
            for j_alim_word_vector in range(len(hist_word_vectors)):
                if len(hist_word_vectors[j_alim_word_vector]):
                    similarity_matrix[i_word_vector, j_alim_word_vector] = self.w.cosine_similarities(accion_word_vectors[i_word_vector], hist_word_vectors[j_alim_word_vector]).mean()



        # Promediado de la matriz de similaridad para normalizar respecto a las distintas acciones posibles
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            similarity_matrix[similarity_matrix==0] = np.nan
            aux_sim_v = np.nanmean(similarity_matrix, axis=0)
            aux_sim_v[np.isnan(aux_sim_v)] = 0

            # Construimos sim_v con la forma que corresponde del tipo (len(historial_v),)
            j=0
            sim_v=[]
            for N in self.N_alim_v:
                sim_v.append(aux_sim_v[j:j+N])
                j += N     


        return np.array(sim_v)


         
    
    
    def googleTrends_sim(self, posible_accion, historial_v, lugar = 'Estados Unidos'):

        # Carga de las keywords de búsqueda con todas las combinaciones de los nombres de alimentos
        # entre los alimentos del historial de comidas y el alimento a ser elegido posiblemente
        keywords = []
        for i in range(len(historial_v)):
            aux_keywords = [posible_accion[:20]]
            for j in range(len(historial_v[i])): aux_keywords += historial_v[i][j][:20]       ## Lo cortamos en 20 caracteres.

            keywords.append(aux_keywords[:])
            
        # Búsqueda de las keywords en la web por el último año en el lugar elegido
        pytrend.build_payload(keywords, cat=0, timeframe='today 12-m', geo = geo_d[lugar])

        # Carga de las tendencias de búsqueda en el tiempo
        trend_v = np.array(pytrend.interest_over_time())

        # Definición de la similaridad entre un alimento y el historial de alimentos del entorno
        sim_v  = trend_v[:,:-1].sum(axis=0)

        return sim_v





    def similaridad_historial(self, posible_accion, historial_v):
        if self.similarity_type is not None:
            if self.similarity_type == 'google_trends':
                sim_v = self.googleTrends_sim(posible_accion,historial_v)
            elif self.similarity_type == 'word_embedding':
                sim_v = self.wordEmb_sim(posible_accion,historial_v)
            else:
                print('\nError! "self.similarity_type" no válido')
                sys.exit()
        else:
            dist_v = cdist(np.expand_dims(posible_accion,axis=0), historial_v, metric = 'euclidean')[0]
            sim_v = 1/dist_v

        return sim_v




    def accion_humano(self, estado, accion_agente):

        hist_v = estado[0]
        acciones_v = estado[1]


        if np.random.random() <= self.prob_recomendacion:
            return accion_agente
        else:
            recompensa_humano = []

            if self.similarity_type is not None:
                espacio_acciones_v = acciones_v['nombres']
                historial_v = hist_v['nombres']
            else:
                espacio_acciones_v = acciones_v['valores']
                historial_v = hist_v['valores']
                
            
            for posible_accion in espacio_acciones_v:
                recompensa_accion_v = self.similaridad_historial(posible_accion, historial_v)*np.exp(self.valoracion_temp*np.arange(len(historial_v), 0, -1))
                recompensa_accion=0
                for recompensa_accion_i in recompensa_accion_v: recompensa_accion += recompensa_accion_i.sum()
                recompensa_humano.append(recompensa_accion)

            return np.argmax(recompensa_humano)





    def calc_recompensa(self, estado, accion_agente):
        # Obtenemos primero la acción efectuada por el humano al elegir el alimento que quiera
        accion_humano = self.accion_humano(estado, accion_agente)

        acciones_v = estado[1]['valores']

        # Buscamos la distancia entre cada alimento posible a elegir y el target nutricional del entorno
        dist_nut_v = cdist(np.expand_dims(self.target_nut,axis=0), acciones_v, metric = 'euclidean')[0]

        recompensa_agente_v = np.zeros(self.N_acciones)

        recompensa_agente = int(np.min(dist_nut_v)/dist_nut_v[accion_humano])   # Hacemos que la recompensa sea solamente 1 o 0

        recompensa_agente_v[accion_agente] = recompensa_agente    # Creamos un vector tipo 'one_hot' para las recompensas del agente en función de las posibles acciones

        return recompensa_agente, accion_humano, recompensa_agente_v

    



    
    

    

        
