import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
from statistics import mean
import random

PRINT = False
DELTA = 21
CSV_INPUT_FOLDER = "../../zip_arrival"
CSV_OUTPUT_FOLDER = "csv_solution"

def get_column_from_df(df, column_name):
    return df[column_name].values.tolist()

def add_column_to_df(df, values_list, new_column_name):
    df[new_column_name] = values_list
    return df

def optimal_solution(T, B, Delta):
    """
    La funzione calcola la soluzione ottima per il problema di pianificazione della campagna di vaccinazione di massa.

    Args:
        T (int): lunghezza dell'intervallo di programmazione della campagna di vaccinazione
        B (list): lista con il numero di dosi di vaccino consegnate ogni giorno dell'intervallo di programmazione
        Delta (int): intervallo di tempo tra la somministrazione della prima e seconda dose di vaccino

    Returns:
        list: lista contenente il numero di seconde dosi somministrate, la quantità di scorti rimanenti al termine del periodo di programmazione ed il valore della soluzione trovata
    """

    if PRINT == True:
        print("\n***** Gurobipy solver - Basic*****\n")
    
    # Crea il modello di ottimizzazione
    m = gp.Model("vaccinations")

    if PRINT == False:
        m.Params.LogToConsole = 0

    # Vengono definite le variabili del modello
    x = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="First_Doses")
    y = m.addVars(T, lb = 0.0, vtype=GRB.CONTINUOUS, name="Second_Doses")
    s = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="Stocks")
    
    # Popola i dizionari
    dict = {}
    for i in range(0, T):
        dict[i] = i+1

    # Crea l'oggetto multidict richiesto dalla libreria 
    time_slot, time  = gp.multidict(dict)

    # Vengono definiti i vincoli del modello
    m.addConstrs( x[i] == y[i+Delta] for i in range(0, T-Delta))
    m.addConstrs( x[i] == 0 for i in range(T-Delta, T))

    m.addConstrs( y[i] == 0 for i in range(0, Delta))

    m.addConstr ( (x[0]  + 0 + s [0] == B[0] + 0))
    m.addConstrs( (x[i] + 0 + s[i] == B[i] + s[i-1] for i in range(1, Delta)))
    m.addConstrs( (x[i] + x[i-Delta] + s[i] == B[i] + s[i-1] for i in range(Delta, T)))

    m.addConstr( (s[T-1] == 0) ) 

    # Viene definita la funzione obiettivo
    m.setObjective( gp.quicksum(y[i] * (i+1) for i in time_slot), GRB.MINIMIZE)

    # Viene eseguita l'ottimizzazione del modello matematico
    m.optimize()

    # Controllo se è stata trovata una soluzione al problema
    if (m.solCount > 0):
        
        # Recupero il valore delle variabili nella soluzione
        resultList = m.getAttr(GRB.Attr.X, m.getVars())
        first_doses_values = resultList[:T]
        second_doses_values = resultList[T:2*T]
        stock_values = resultList[2*T:]
        object_function_value = m.objVal
        
        if PRINT == True:
            m.printAttr("X")
            print("\n***** Verbose solution printing *****\n")
            print("\n***** Solution's values list printing *****\n")
            print("First_doses values: " + str(first_doses_values))
            print("Second_doses values: " + str(second_doses_values))
            print("Stocks values: " + str(stock_values))
            print("Object_function value: " + str(object_function_value))
                
        return [second_doses_values, stock_values, object_function_value]

    else:
        if PRINT == True:
            print("\n***** No solutions found *****")

def heuristic(T, B, Delta):
    """
    La funzione calcola la soluzione ottenuta con l'utilizzo dell'algoritmo euristico.

    Args:
        T (int): lunghezza dell'intervallo di programmazione della campagna di vaccinazione
        B (list): lista con il numero di dosi di vaccino consegnate ogni giorno dell'intervallo di programmazione
        Delta (int): intervallo di tempo tra la somministrazione della prima e seconda dose di vaccino

    Returns:
        list: lista contenente il valore della soluzione trovata
    """

    x = [0]*T
    y = [0]*T
    s = [0]*T
    object_function_value = 0

    x[T-Delta-1] = int(sum(B)/2)
    y[T-1] = x[T-Delta-1]

    for i in range(0, T):
        if i < T-Delta-1:
            s[i] = sum(B[:i+1])    
        elif i >= T-Delta-1:
            s[i] = sum(B[:i+1]) - x[T-Delta-1]

    s[T-2] = x[T-Delta-1] - B[T-1]

    t = T-Delta-1

    while(t > -1):
        c = min(s[t+Delta-1], x[t], sum(B[:t]))
        
        x[t-1], y[t+Delta-1], x[t], y[t+Delta] = x[t-1]+c, y[t+Delta-1]+c, x[t]-c, y[t+Delta]-c
        s[t-1], s[t+Delta-1] = s[t-1]-c, s[t+Delta-1]-c
       
        t -= 1

    object_function_value = sum(x[j]*(Delta+j+1) for j in range(T))

    return [object_function_value]
    
if __name__ == "__main__":
    
    optimal_result = []
    heuristic_result = []
    result_difference = []
    instances = []
    avg_heuristic = []
    avg_optimal = []
    avg_result_difference = []

    for i in os.listdir(CSV_INPUT_FOLDER):

        print("Processing instance: " + i)
        
        data = pd.read_csv(CSV_INPUT_FOLDER + "/" + i,  index_col=0) 
        b_list = get_column_from_df(data, "ndosi")
        result = optimal_solution(len(b_list), b_list, DELTA)

        data = add_column_to_df(data, result[0], "second_doses")
        data = add_column_to_df(data, result[1], "stock_values")
        optimal_result.append(round( result[2]/(sum(b_list)/2), 4))
        
        heu_result = heuristic(len(b_list), b_list, DELTA)

        heuristic_result.append(round(heu_result[0]/(sum(b_list)/2), 4))
        result_difference.append(round((heu_result[0]-result[2])/(result[2]) * 100, 4))
        
        data.to_csv(CSV_OUTPUT_FOLDER + "/solution_" + i + ".csv")
        
    for j in range(0, len(os.listdir(CSV_INPUT_FOLDER))):
        instances.append(j)
        avg_heuristic.append( round( heuristic_result[j], 4) )
        avg_optimal.append( round( optimal_result[j], 4) )
        avg_result_difference.append( round( result_difference[j], 4) )

    df = pd.DataFrame(instances, columns= ['instance'])
    df['LP Model value'] = avg_optimal
    df['Heuristic'] = avg_heuristic
    df['Avg result difference (%)'] = avg_result_difference

    df.to_csv("result_summary.csv", index=0)