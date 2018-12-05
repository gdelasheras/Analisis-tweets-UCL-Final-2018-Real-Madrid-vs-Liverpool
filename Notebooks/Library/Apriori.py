import pandas as pd
import numpy as np
from IPython.display import clear_output

class APriori():
    """
    Clase con la funcionalidad para ejecutar el algoritmo apriori para hallar patrones de asociación.
    """
    def ExcelPrueba(self, Fichero):
        """
        Función para cargar el fichero de pruebas "dummy".
        :param Fichero: Ruta del fichero.
        """
        fichero_datos = pd.ExcelFile(Fichero)
        fichero_datos.sheet_names
        self.Datos = fichero_datos.parse("Hoja1")
        self.Datos = self.Datos.groupby('id')['trn'].apply(list)
        self.Datos = self.Datos.reset_index(drop=False)
        self.Datos["items"] = self.Datos['trn'].apply(lambda x: self.__UniquePrueba(x))
        self.Itemset = self.__CalcularItemset(self.Datos['items'])
        self.Itemset = np.array(sorted(self.Itemset))

    def Carga(self, Datos, Columna):
        """
        Función para cargar un fichero de datos con los patrones ya agrupados.
        :param Datos: Ruta del fichero de datos.
        :param Columna: Columna donde se encuentran los datos.
        """
        self.Datos = Datos
        self.Datos["items"] = self.Datos[Columna].apply(lambda x: self.__Unique(x))
        self.Itemset = self.__CalcularItemset(self.Datos['items'])
        self.Itemset = np.array(sorted(self.Itemset))

    def __Combinaciones(self, Columna, K):
        """
        Función para calcular las combinaciones de los itemsets candidatos.
        :param Columna: Columna donde están los itemsets.
        :param K: Iteración.
        """
        combinaciones = []
        for item in Columna:
            for item_2 in Columna:
                if item is not item_2:
                    for elemento in item_2:
                        temp = item.copy()
                        if elemento not in item:
                            temp.append(elemento)
                        temp = sorted(temp)
                        if temp not in combinaciones:
                            if len(temp) == K:
                                combinaciones.append(temp)
        return combinaciones

    def __UniquePrueba(self, Array):
        """
        Función para calcular el itemset de los ficheros de prueba.
        :param Array: Array de items.
        """
        return np.unique(Array[0].split(','))

    def __Unique(self, Array):
        """
        Función para calcular el itemset de los ficheros.
        :param Array: Array de items.
        """
        return np.unique(Array)

    def __CalcularFreqSoporteRefractor(self, Item, Columna):
        """
        Función para calcular la frecuencia soporte.
        :param Item: Item sobre el que se va a calcular.
        :param Columna: Columna del dataset para calcular la frecuencia soporte.
        :return: Devuelve la frecuencia soporte.
        """
        count = 0
        for fila in Columna:
            result = 0
            for item_i in Item:
                if item_i in fila:
                    result += 1
            if result == len(Item):
                count += 1
        return count

    def __CalcularItemset(self, Columna):
        """
        Función para calcular los itemsets.
        :param Columna: Columna con las transacciones.
        :return: Devuelve la lista de itemsets.
        """
        lista_itemset = []
        for i in range(0, len(Columna)):
            print("Calculando itemsets...")
            print("itemset", i, "de", len(Columna))
            clear_output(wait=True)
            for item in Columna[i]:
                if item not in lista_itemset:
                    lista_itemset.append(item)
        return lista_itemset

    def __CalcularItemSetsFrecuentes(self, MinimoSoporte=None, MinimoFreqSop=None):
        """
        Función para calcular los itemsets frecuentes.
        :param MinimoSoporte: Soporte umbral.
        :param MinimoFreqSop: Frecuencia soporte umbral.
        """
        # K = 1
        print("")
        print("Iniciando algoritmo apriori para patrones de asociación...")
        print("")
        print("-------")
        print("")
        print("Probando k = 1")

        self.Soporte = pd.DataFrame()
        self.Soporte["Item"] = self.Itemset
        print("")
        print("Calculando items...")
        self.Soporte["Item"] = self.Soporte['Item'].apply(lambda x: [x])
        print("")
        print("Calculando Frec.Soporte...")
        self.Soporte["Frec. Soporte"] = self.Soporte['Item'].apply(lambda x: self.__CalcularFreqSoporteRefractor(x, self.Datos["items"]))
        print("")
        print("Calculando Soporte...")
        self.Soporte["Soporte"] = self.Soporte["Frec. Soporte"] / len(self.Datos)

        if MinimoSoporte is not None:
            self.Soporte = self.Soporte[self.Soporte["Soporte"] >= MinimoSoporte]
        elif MinimoFreqSop is not None:
            self.Soporte = self.Soporte[self.Soporte["Frec. Soporte"] >= MinimoFreqSop]

        End = False
        k = 2

        display(self.Soporte)

        print("")
        print("-------")
        print("")

        # Iteramos hasta que no queden secuencias candidatas.
        while End is False:
            print("Probando k = " + str(k))

            Soporte_bk = self.Soporte.copy()
            Dataframe_temp = pd.DataFrame()
            print("Calculando combinaciones")
            Dataframe_temp["Item"] = self.__Combinaciones(Soporte_bk["Item"], k)
            print("Calculando Frec.Soporte")
            Dataframe_temp["Frec. Soporte"] = Dataframe_temp['Item'].apply(lambda x: self.__CalcularFreqSoporteRefractor(x, self.Datos["items"]))
            print("Calculando Soporte")
            Dataframe_temp["Soporte"] = Dataframe_temp["Frec. Soporte"] / len(self.Datos)

            if MinimoSoporte is not None:
                print("Filtrando Soporte mínimo >= " + str(MinimoSoporte))
                Dataframe_temp = Dataframe_temp[Dataframe_temp["Soporte"] >= MinimoSoporte]
            elif MinimoFreqSop is not None:
                print("Filtrando Frec. Soporte mínimo >= " + str(MinimoFreqSop))
                Dataframe_temp = Dataframe_temp[Dataframe_temp["Frec. Soporte"] >= MinimoFreqSop]

            if len(Dataframe_temp) is not 0:
                self.Soporte = Dataframe_temp.copy()
                self.Soporte = self.Soporte.reset_index(drop=False)
                del self.Soporte["index"]
                k = k + 1
                display(self.Soporte.head(10))

            elif k == 2:
                print("Terminado")
                End = True
                Soporte_bk = Soporte_bk.iloc[0:0]

            else:
                print("Terminado")
                End = True

            print("")
            print("-------")
            print("")


        self.Reglas = Soporte_bk

    def __ExtraerReglas(self, ConfianzaMinima):
        """
        Función para extraer las reglas con una confianza mínima.
        :param ConfianzaMinima: Confianza mínima de las reglas.
        """
        self.ReglasConfianza = pd.DataFrame(columns=["r_1", "r_2"])
        arr_r_1 = []
        arr_r_2 = []

        soporte_r_1 = []

        for idx_fila in range(0, len(self.Reglas)):
            regla = self.Reglas.loc[idx_fila, "Item"]

            for idx_regla in range(0, len(regla)):
                regla_tmp = regla.copy()
                r_1 = regla_tmp[idx_regla]
                del regla_tmp[idx_regla]

                soporte_r_1.append(self.Reglas.loc[idx_fila, "Frec. Soporte"])
                arr_r_1.append(np.array(r_1).ravel())
                arr_r_2.append(np.array(regla_tmp).ravel())

                soporte_r_1.append(self.Reglas.loc[idx_fila, "Frec. Soporte"])
                arr_r_2.append(np.array(r_1).ravel())
                arr_r_1.append(np.array(regla_tmp).ravel())

        self.ReglasConfianza["r_1"] = arr_r_1
        self.ReglasConfianza["r_2"] = arr_r_2
        self.ReglasConfianza["soporte_r_1"] = soporte_r_1
        self.ReglasConfianza["soporte_r_2"] = self.ReglasConfianza["r_1"].apply(
            lambda x: self.__CalcularFreqSoporteRefractor(x, self.Datos["items"]))

        self.ReglasConfianza["confianza"] = round((self.ReglasConfianza["soporte_r_1"] / self.ReglasConfianza["soporte_r_2"]) * 100, 0)
        self.ReglasConfianza["temp"] = self.ReglasConfianza["r_1"].apply(lambda x: ''.join(x))
        self.ReglasConfianza = self.ReglasConfianza.sort_values(["temp"])
        self.ReglasConfianza["temp"] += self.ReglasConfianza["r_2"].apply(lambda x: ''.join(x))
        self.ReglasConfianza.drop_duplicates(subset='temp', keep="last")
        del self.ReglasConfianza["temp"]
        self.ReglasConfianza = self.ReglasConfianza.reset_index(drop=True)
        self.ReglasConfianza= self.ReglasConfianza[self.ReglasConfianza["confianza"] >= ConfianzaMinima]

        self.ReglasConfianza = self.ReglasConfianza.sort_values(by=['confianza'], ascending=False)
        self.ReglasConfianza = self.ReglasConfianza.reset_index(drop=True)

        print("Reglas de asociación: ")

        display(self.ReglasConfianza)

    def CalcularReglasDeConfianza(self, Confianza, MinimoSoporte=None, MinimoFreqSop=None, Echo=False):
        """
        Función principal para calculas la reglas
        :param Confianza: Confianza mínima de las reglas.
        :param MinimoSoporte: Soporte mínimo de las reglas.
        :param MinimoFreqSop: Frecuencia soporte mínima de las reglas.
        :param Echo: Indica si se imprime por pantalla el avance de la función.
        """
        # Se calculan los itemsets
        if MinimoSoporte is not None:
            self.__CalcularItemSetsFrecuentes(MinimoSoporte=MinimoSoporte)
        elif MinimoFreqSop is not None:
            self.__CalcularItemSetsFrecuentes(MinimoFreqSop=MinimoFreqSop)

        # Se extraen las reglas
        self.__ExtraerReglas(Confianza)