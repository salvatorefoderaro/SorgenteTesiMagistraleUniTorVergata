import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
from pathlib import Path
from statistics import mean
import shutil
import numpy as np

PRINT = False
DELTA = {'Pfizer': 21, 'Moderna': 28, 'Astrazeneca': 78}
CSV_INPUT_FOLDER = "../../zip_arrival"
CSV_OUTPUT_FOLDER = "csv_solution"
T = 180
NUMBER_OF_ELEMENT = 1000
LIMIT_CAPACITY = 3
INCREMENT = 1
LAST_DAY_VACCINES = True
ALPHA_LIST = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]

def optimize_test_capacity_multiple_vaccines(T, B, Delta, Capacity, heu_result, alpha):

    """
    Ottimizza la capacità di test per la pianificazione delle vaccinazioni utilizzando più vaccini

    Args:
        T (int): lunghezza dell'intervallo di programmazione
        B (dict): dizionario contenente il numero di dosi consegnate, ogni giorno, per ogni vaccino
        Delta (dict): dizionario contenente, per ogni vaccino, l'intervallo di tempo tra la somministrazione della prima e seconda dose 
        Capacity (int): la capacità massima di somministrazione giornaliera
        alpha (float): valore del parametro alpha considerato per la risoluzione del problema

    Returns:
        list: una lista contenete l'indicazione per ogni giorno, del numero di prime e seconde dosi da somministrare, il numero di scorte al termine, il valore della funzione obiettivo ed il valore del carico massimo di lavoro a cui è sottoposto lo staff
    """

    # Crea il modello di ottimizzazione
    m = gp.Model("vaccinations")

    m.Params.LogToConsole = 0
    
    # Crea i dizionari richiesti da gurobipy
    dict = {}
    dict_B = {}

    # Popola i dizionari
    for i in B:
        for j in range(0, T):
            dict[(i, j)] = [j+1] # dict[("Pfizer", 0)] = 1, ...
            dict_B[(i,j)] = B[i][j]  # dict_B[("Pfizer", 0)] = 343243, ...

    # Salva il valore originale di B e sostituisce con arrival_dict
    original_B = B
    B = dict_B

    # Crea l'oggetto multidict richiesto dalla libreria 
    combinations, time_frame = gp.multidict(dict)

    # Vengono definite le variabili del modello
    x = m.addVars(combinations, lb=0.0, vtype=GRB.INTEGER, name="First_Doses")
    y = m.addVars(combinations, lb = 0.0, vtype=GRB.INTEGER, name="Second_Doses")
    s = m.addVars(combinations, lb=0.0, vtype=GRB.INTEGER, name="Stocks")
    maximum_workload = m.addVar(lb = 0.0, vtype = GRB.INTEGER, name="Z")

    # Vengono definiti i vincoli del modello
    m.addConstrs( (x.sum('*',j) + y.sum('*', j) <= Capacity for j in range(0, T)))  

    m.addConstrs( (x[j, i] == y[j,i+Delta[j]] for j, i in combinations if i < T-Delta[j] ))
    m.addConstrs( (x[j, i] == 0 for j, i in combinations if i >= T-Delta[j] ))

    m.addConstrs( y[j,i] == 0 for j, i in combinations if i < Delta[j])
    
    m.addConstrs( (x[j,i]  + 0 + s[j,i] == B[j,i] + 0 for j, i in combinations if i == 0))
    m.addConstrs( (x[j,i] + 0 + s[j,i] == B[j,i] + s[j,i-1] for j, i in combinations if i >= 1 and i < Delta[j]))
    m.addConstrs( (x[j,i] + x[j,i-Delta[j]] + s[j,i] == B[j,i] + s[j, i-1] for j, i in combinations if i >= 1 and i >= Delta[j]))

    for i in range(0, 180):
            m.addConstr( maximum_workload >= x["Astrazeneca",i] + x["Moderna",i] + x["Pfizer",i] + y["Astrazeneca",i] + y["Moderna",i] + y["Pfizer",i])

    # Viene definita la funzione obiettivo
    m.setObjective( alpha * (maximum_workload) + (1-alpha) * ( ( ( gp.quicksum(y[j,i] * i for j,i in combinations ) + s["Pfizer", T-1] * (180 + Delta["Pfizer"]) + s["Moderna", T-1] * (180 + Delta["Moderna"]) + s["Astrazeneca", T-1] * (180 + Delta["Astrazeneca"]) ) / (1) ) / 1)  + 1000 * (s["Astrazeneca", T-1] + s["Moderna", T-1] + s["Pfizer", T-1]) , GRB.MINIMIZE)
   
    # Viene eseguita l'ottimizzazione del modello matematico
    m.optimize()
   
    # Controllo se è stata trovata una soluzione al problema    
    if (m.solCount > 0):

        # Recupero il valore delle variabili nella soluzione
        resultList = m.getAttr(GRB.Attr.X, m.getVars())
            
        # Popolo i dizionari per l'output dei risultati
        first_doses_dict = {}
        second_doses_dict = {}
        stocks_dict = {}

        for i in range(0, len(original_B)):
            first_doses_dict[list(original_B)[i]] = resultList[T*i:T*(i+1)]

        for i in range(0, len(original_B)):
            second_doses_dict[list(original_B)[i]] = resultList[T*(i+len(original_B)):T*(i+1+len(original_B))]
        
        for i in range(0, len(original_B)):
            stocks_dict[list(original_B)[i]] = resultList[T*(i+2*len(original_B)):T*(i+1+2*len(original_B))]

        object_function_value = m.objVal

        return[first_doses_dict, second_doses_dict, stocks_dict, object_function_value, maximum_workload.X]
    else:
        print("\n***** No solutions found *****")

def get_column_from_df(df, column_name):
    """
    La funzione permette di estrarre i valori di una specifica colonna all'interno di un DataFrame e convertirli in lista.

    Args:
        df (DataFrame): 
        column_name (string): nome della colonna nel DataFramelista con il numero di dosi di vaccino consegnate ogni giorno dell'intervallo di programmazione
        
    Returns:
        list: 
    """
    return df[column_name].values.tolist()

def add_column_to_df(df, values_list, new_column_name):
    """
    La funzione permette, partendo da una lista di valori, di aggiungere una nuova colonna all'interno di un DataFrame.

    Args:
        df (DataFrame):
        values_list (list): lista di valori da inserire come nuova colonna all'interno del DataFrame 
        new_column_name (string): nome della colonna nel DataFramelista con il numero di dosi di vaccino consegnate ogni giorno dell'intervallo di programmazione
        
    Returns:
        DataFrame: nuovo DataFrame ottenuto con l'inserimento della nuova colonna 
    """
    df[new_column_name] = values_list
    return df

def heuristic_v2_sum(first_doses_list, delta, t, q):
    """
    La funzione restituisce il numero di seconde dosi la cui somministrazione è già programmata per i successivi q giorni.

    Args:
        first_doses_list (): lista contenente la quantità di prime dosi somministrate per ogni giorno
        delta (int): intervallo tra la somministrazione della prima e seconda dose
        t (int): indicazione del giorno di partenza per il quale eseguire il calcolo
        q (int): numero di giorni per cui è necessario garantire la somministrazione delle seconde dosi già programmate

    Returns:
        list: 
    """

    value_sum = 0

    for i in range(1, q+1):
        if t + i - delta >= 0 and t + i - delta <= 180:
            value_sum += first_doses_list[t + i - delta]
    return value_sum

def heuristic_v2_risk(arrival_dict, capacity, q):
    """
    Implementazione dell'algoritmo q-days-ahead.

    Args:
        arrival_dict (dict): dizionario contenente, per ogni vaccino, la lista con l'indicazione, per ogni giorno, del numero di dosi consegnate
        capacity (int): capacità massima di somministrazione giornaliera
        q (int): numero di giorni per cui è necessario garantire la somministrazione delle seconde dosi già programmate

    Returns:
        list: 
    """

    vaccines = { "Pfizer": [21, [0]*180, [0]*180], "Moderna": [28, [0]*180, [0]*180], "Astrazeneca": [78, [0]*180, [0]*180] }
    total_arrival = {"Pfizer": sum(arrival_dict["Pfizer"]), "Moderna": sum(arrival_dict["Moderna"]), "Astrazeneca": sum(arrival_dict["Astrazeneca"])   }
    doses_somministrated = {"Pfizer": 0, "Moderna": 0, "Astrazeneca": 0}
    arrival_sum = total_arrival["Pfizer"] + total_arrival["Moderna"] + total_arrival["Astrazeneca"]

    sum_total_arrival = total_arrival["Pfizer"] + total_arrival["Moderna"] + total_arrival["Astrazeneca"]
    object_function = 0 
    count_negative = 0
    negative_stocks = 0
    utilization = 0
    min_negative_stocks = 0
    x_cumulative =  0
    maximum_workload = 0
    stocks_at_end = 0

    for t in range(0, 180):
        for vaccine_name in vaccines:

            x_planned = 0
            y_planned = 0
            delta = vaccines[vaccine_name][0]
            x = vaccines[vaccine_name][1]
            s = vaccines[vaccine_name][2]
            arrival_list = arrival_dict[vaccine_name]

            if t + delta < 180:

                if t == 0:
                    x_planned = arrival_list[t]
                elif t-delta < 0:
                    x_planned = s[t-1] + arrival_list[t]
                else:
                    x_planned = s[t-1] - x[t-delta] - heuristic_v2_sum(x, delta, t, q) + arrival_list[t]

                if x_planned < 0:
                    x_planned = 0

                if t-delta >= 0:
                    y_planned = x[t-delta]
                else:
                    y_planned = 0

                x_planned = min(x_planned , (capacity * 0.33) - y_planned)

                object_function += (t+1+delta)*x_planned
                utilization += x_planned
                doses_somministrated[vaccine_name] += x_planned
                vaccines[vaccine_name][1][t] = x_planned

                if t == 0:
                    vaccines[vaccine_name][2][t] = - x_planned  + arrival_list[t]
                elif t - delta < 0:
                    vaccines[vaccine_name][2][t] = s[t-1] - x_planned + arrival_list[t]
                else:
                    vaccines[vaccine_name][2][t] = s[t-1] - x_planned - x[t-delta] + arrival_list[t]
            else:
                vaccines[vaccine_name][2][t] = s[t-1]  - x[t-delta]  + arrival_list[t]

            if vaccines[vaccine_name][2][t] < 0:
                negative_stocks += vaccines[vaccine_name][2][t]
                count_negative += 1
                min_negative_stocks = min(min_negative_stocks, vaccines[vaccine_name][2][t])

    x_cumulative += doses_somministrated["Pfizer"] * 2
    x_cumulative += doses_somministrated["Astrazeneca"] * 2
    x_cumulative += doses_somministrated["Moderna"] * 2

    for tt in range(0, 180):
        total_x_y_day = vaccines["Pfizer"][1][tt] + vaccines["Pfizer"][2][tt] + vaccines["Moderna"][1][tt] + vaccines["Moderna"][2][tt] + vaccines["Astrazeneca"][1][tt] + vaccines["Astrazeneca"][2][tt]
        if total_x_y_day > maximum_workload:
            maximum_workload = total_x_y_day

    stocks_at_end += total_arrival["Pfizer"] - doses_somministrated["Pfizer"] * 2
    stocks_at_end += total_arrival["Moderna"] - doses_somministrated["Moderna"] * 2
    stocks_at_end += total_arrival["Astrazeneca"] - doses_somministrated["Astrazeneca"] * 2

    if LAST_DAY_VACCINES:
        object_function += (total_arrival["Pfizer"] - doses_somministrated["Pfizer"] * 2) * (180+21)
        object_function += (total_arrival["Moderna"] - doses_somministrated["Moderna"] * 2) * (180+28)
        object_function += (total_arrival["Astrazeneca"] - doses_somministrated["Astrazeneca"] * 2) * (180+78)
        return [ (object_function ) / ((x_cumulative + stocks_at_end)/2) , 0, abs(negative_stocks)/arrival_sum, count_negative/ (3*180), (utilization)/sum_total_arrival, min_negative_stocks, maximum_workload]

    else:
        return [ (object_function ) / (x_cumulative) , stocks_at_end, abs(negative_stocks)/arrival_sum, (count_negative)/ (3*180), (utilization)/sum_total_arrival, min_negative_stocks]

def heuristic_v2(arrival_dict, capacity):
    """
    Implementazione dell'algoritmo conservativo.

    Args:
        arrival_dict (dict): dizionario contenente, per ogni vaccino, la lista con l'indicazione, per ogni giorno, del numero di dosi consegnate
        capacity (int): capacità massima di somministrazione giornaliera

    Returns:
        list: 
    """
    vaccines = { "Pfizer": [21, [0]*180, [0]*180], "Moderna": [28, [0]*180, [0]*180], "Astrazeneca": [78, [0]*180, [0]*180] }
    s = []
    total_arrival = {"Pfizer": sum(arrival_dict["Pfizer"]), "Moderna": sum(arrival_dict["Moderna"]), "Astrazeneca": sum(arrival_dict["Astrazeneca"])   }
    doses_somministrated = {"Pfizer": 0, "Moderna": 0, "Astrazeneca": 0}
    sum_total_arrival = total_arrival["Pfizer"] + total_arrival["Moderna"] + total_arrival["Astrazeneca"]

    object_function = 0 
    y_cumulative = 0
    x_cumulative =  0
    maximum_workload = 0
    stocks_at_end = 0

    for t in range(0, 180):
        
        for vaccine_name in vaccines:
            delta = vaccines[vaccine_name][0]
            x = vaccines[vaccine_name][1]
            s = vaccines[vaccine_name][2]
            arrival_list = arrival_dict[vaccine_name]

            if t + delta < 180:

                if t == 0:
                    x_planned = arrival_list[t]
                elif t-delta < 0:
                    x_planned = s[t-1] + arrival_list[t]
                else:
                    x_planned = s[t-1] - x[t-delta] + arrival_list[t]

                x_planned = int(x_planned / 2)

                if t-delta >= 0:
                    y_planned = x[t-delta]
                else:
                    y_planned = 0
    
                x_planned = min(x_planned, (capacity * 0.33) - y_planned)

                y_cumulative += x_planned
                object_function += (t+1+delta)*x_planned
                vaccines[vaccine_name][1][t] = x_planned
                doses_somministrated[vaccine_name] += x_planned
            
                if t == 0:
                    vaccines[vaccine_name][2][t] = - x_planned  + arrival_list[t]
                elif t - delta < 0:
                    vaccines[vaccine_name][2][t] = s[t-1] - x_planned + arrival_list[t]
                else:
                    vaccines[vaccine_name][2][t] = s[t-1] - x_planned - x[t-delta] + arrival_list[t]
            else:
                vaccines[vaccine_name][2][t] = s[t-1]  - x[t-delta]  + arrival_list[t]

    x_cumulative += doses_somministrated["Pfizer"] * 2
    x_cumulative += doses_somministrated["Astrazeneca"] * 2
    x_cumulative += doses_somministrated["Moderna"] * 2

    for tt in range(0, 180):
        total_x_y_day = vaccines["Pfizer"][1][tt] + vaccines["Pfizer"][2][tt] + vaccines["Moderna"][1][tt] + vaccines["Moderna"][2][tt] + vaccines["Astrazeneca"][1][tt] + vaccines["Astrazeneca"][2][tt]
        if total_x_y_day > maximum_workload:
            maximum_workload = total_x_y_day

    stocks_at_end += total_arrival["Pfizer"] - doses_somministrated["Pfizer"] * 2
    stocks_at_end += total_arrival["Moderna"] - doses_somministrated["Moderna"] * 2
    stocks_at_end += total_arrival["Astrazeneca"] - doses_somministrated["Astrazeneca"] * 2

    if stocks_at_end - (vaccines["Astrazeneca"][2][179] +  vaccines["Pfizer"][2][179] +  vaccines["Moderna"][2][179]) > 100:
        print("Errore")
        input()

    if LAST_DAY_VACCINES:

        object_function += (total_arrival["Pfizer"] - doses_somministrated["Pfizer"] * 2)  * (180+21)
        object_function += (total_arrival["Moderna"] - doses_somministrated["Moderna"] * 2)  * (180+28)
        object_function += (total_arrival["Astrazeneca"] - doses_somministrated["Astrazeneca"] * 2)  * (180+78)

        return [ (object_function ) / ((x_cumulative + stocks_at_end)/2) , 0, (x_cumulative/2)/sum_total_arrival, maximum_workload]
    
    else:
        return [ (object_function ) / (x_cumulative) , stocks_at_end, (x_cumulative/2)/sum_total_arrival]

if __name__ == "__main__":

    shutil.rmtree(CSV_OUTPUT_FOLDER)
    os.mkdir(CSV_OUTPUT_FOLDER)

    penality_optimal_result = {}
    optimal_result = {}
    optimal_result_z = {}

    min_negative_stocks_risk = {}
    min_negative_stocks_risk_7 = {}
    min_negative_stocks_risk_14 = {}
    min_negative_stocks_risk_21 = {}

    feasible_s_pfizer = 0
    feasible_s_moderna = 0
    feasible_s_astrazeneca = 0
    
    heuristic_result = {}
    heuristic_risk_result = {}
    heuristic_risk_result_7 = {}
    heuristic_risk_result_14 = {}
    heuristic_risk_result_21 = {}

    heuristic_result_z = {}
    heuristic_risk_result_z = {}
    heuristic_risk_result_7_z = {}
    heuristic_risk_result_14_z = {}
    heuristic_risk_result_21_z = {}

    utilization_T = {}
    utilization_T_heuristic = {}
    utilization_T_heuristic_risk = {}
    utilization_T_heuristic_risk_7 = {}
    utilization_T_heuristic_risk_14 = {}
    utilization_T_heuristic_risk_21 = {}

    penality_heuristic = {}
    penality_heuristic_risk = {}
    penality_heuristic_risk_7 = {}
    penality_heuristic_risk_14 = {}
    penality_heuristic_risk_21 = {}

    heuristic_risk_count_negative = {}
    heuristic_risk_count_negative_7 = {}
    heuristic_risk_count_negative_14 = {}
    heuristic_risk_count_negative_21 = {}

    heuristic_risk_negative_arrival = {}
    heuristic_risk_negative_arrival_7 = {}
    heuristic_risk_negative_arrival_14 = {}
    heuristic_risk_negative_arrival_21 = {}

    z_value = []

    num = 1
    while(num < (LIMIT_CAPACITY)):
        penality_optimal_result[str(num) + " c"] = {}
        optimal_result[str(num) + " c"] = {}
        optimal_result_z[str(num) + " c"] = {}
        utilization_T[str(num) + " c"] = {}

        for alpha_value in ALPHA_LIST:
            penality_optimal_result[str(num) + " c"][alpha_value] = []
            optimal_result[str(num) + " c"][alpha_value] = []
            optimal_result_z[str(num) + " c"][alpha_value] = []
            utilization_T[str(num) + " c"][alpha_value] = []

        min_negative_stocks_risk[str(num) + " c"] = 0
        min_negative_stocks_risk_7[str(num) + " c"] = 0
        min_negative_stocks_risk_14[str(num) + " c"] = 0
        min_negative_stocks_risk_21[str(num) + " c"] = 0
        
        heuristic_result[str(num) + " c"] = []
        heuristic_risk_result[str(num) + " c"] = []
        heuristic_risk_result_7[str(num) + " c"] = []
        heuristic_risk_result_14[str(num) + " c"] = []
        heuristic_risk_result_21[str(num) + " c"] = []

        heuristic_result_z[str(num) + " c"] = []
        heuristic_risk_result_z[str(num) + " c"] = []
        heuristic_risk_result_7_z[str(num) + " c"] = []
        heuristic_risk_result_14_z[str(num) + " c"] = []
        heuristic_risk_result_21_z[str(num) + " c"] = []

        penality_heuristic[str(num) + " c"] = []
        penality_heuristic_risk[str(num) + " c"] = []
        penality_heuristic_risk_7[str(num) + " c"] = []
        penality_heuristic_risk_14[str(num) + " c"] = []
        penality_heuristic_risk_21[str(num) + " c"] = []

        utilization_T_heuristic[str(num) + " c"] = []
        utilization_T_heuristic_risk[str(num) + " c"] = []
        utilization_T_heuristic_risk_7[str(num) + " c"] = []
        utilization_T_heuristic_risk_14[str(num) + " c"] = []
        utilization_T_heuristic_risk_21[str(num) + " c"] = []

        heuristic_risk_count_negative[str(num) + " c"] = []
        heuristic_risk_count_negative_7[str(num) + " c"] = []
        heuristic_risk_count_negative_14[str(num) + " c"] = []
        heuristic_risk_count_negative_21[str(num) + " c"] = []

        heuristic_risk_negative_arrival[str(num) + " c"] = []
        heuristic_risk_negative_arrival_7[str(num) + " c"] = []
        heuristic_risk_negative_arrival_14[str(num) + " c"] = []
        heuristic_risk_negative_arrival_21[str(num) + " c"] = []

        num += INCREMENT
        num = round(num, 2)

    file_list = os.listdir(CSV_INPUT_FOLDER)
    instances = np.arange(1, T + 1, 1).tolist()

    for instance_number in range(0, len(os.listdir(CSV_INPUT_FOLDER)) - (1000 - NUMBER_OF_ELEMENT)):

        print("Processing instance: " + str(instance_number))

        df = pd.DataFrame(instances, columns= ['instance'])

        # Read the csv files
        data = pd.read_csv(CSV_INPUT_FOLDER + "/" + file_list[instance_number],  index_col=0) 
        data_1 = pd.read_csv(CSV_INPUT_FOLDER + "/" + file_list[instance_number+1],  index_col=0) 
        data_2 = pd.read_csv(CSV_INPUT_FOLDER + "/" + file_list[instance_number+2],  index_col=0) 

        # Get column from the csv files
        b_list_0 = get_column_from_df(data, "ndosi")
        b_list_1 = get_column_from_df(data_1, "ndosi")
        b_list_2 = get_column_from_df(data_2, "ndosi")
        df["arrival_pfizer"] = b_list_0
        df["arrival_moderna"] = b_list_1
        df["arrival_astrazeneca"] = b_list_2

        arrival_dict = {'Pfizer': b_list_0, 'Moderna': b_list_1, 'Astrazeneca': b_list_2}
        
        # Build the dict with different capacity
        total_capacity = sum(b_list_0) + sum(b_list_1) + sum(b_list_2)
        capacity = {}

        num = 1
        while(num < LIMIT_CAPACITY):
            capacity[str(num) + " c"] =  int(num * int(total_capacity/180))
            num += INCREMENT
            num = round(num, 2)

        feasible_s_pfizer = max(feasible_s_pfizer, min(0, sum(b_list_0) - 2*sum(b_list_0[:180-21])))
        feasible_s_moderna = max(feasible_s_moderna, min(0, sum(b_list_1) - 2*sum(b_list_1[:180-28])))
        feasible_s_astrazeneca = max(feasible_s_astrazeneca, min(0, sum(b_list_2) - 2*sum(b_list_2[:180-78])))

        # For each capacity...
        for vaccine_name in capacity:

            df["capacity"] = [capacity[vaccine_name]] * 180

            heu_result = heuristic_v2(arrival_dict, capacity[vaccine_name])
            heu_result_risk = heuristic_v2_risk(arrival_dict, capacity[vaccine_name], 1)
            heu_result_risk_7 = heuristic_v2_risk(arrival_dict, capacity[vaccine_name], 7)
            heu_result_risk_14 = heuristic_v2_risk(arrival_dict, capacity[vaccine_name], 14)
            heu_result_risk_21 = heuristic_v2_risk(arrival_dict, capacity[vaccine_name], 21)
            
            for alpha_value in ALPHA_LIST:
                #####
                result = optimize_test_capacity_multiple_vaccines(len(b_list_0), arrival_dict, DELTA, capacity[vaccine_name], heu_result[0], alpha_value)
                second_doses_sum = 0
                penality_sum = 0
                new_doses = 0
                solution = 0

                for j in result[0]:
                    df["first_doses_" + j] = result[0][j]
                    df["second_doses_" + j] = result[1][j]
                    df["stock_values_" + j] = result[2][j]
                    second_doses_sum += sum(result[1][j])
                    penality_sum += result[2][j][180-1]
                    new_doses += (result[2][j][180-1]/2) * (180 + DELTA[j])
                    for nn in range(0, len(result[1][j])):
                        solution += result[1][j][nn] * nn
                
                # Calculate the optimal result and append to the CSV file
                optimal_result_z[vaccine_name][alpha_value].append(result[4])
                optimal_result_1 = (solution + new_doses) / (second_doses_sum + penality_sum/2)
                optimal_result[vaccine_name][alpha_value].append( optimal_result_1 )
                penality_optimal_result[vaccine_name][alpha_value].append(0)
                utilization_T[vaccine_name][alpha_value].append( second_doses_sum / total_capacity )
                #####

            min_negative_stocks_risk[vaccine_name] = min(min_negative_stocks_risk[vaccine_name], heu_result_risk[5])
            min_negative_stocks_risk_7[vaccine_name] = min(min_negative_stocks_risk_7[vaccine_name], heu_result_risk_7[5])
            min_negative_stocks_risk_14[vaccine_name] = min(min_negative_stocks_risk_14[vaccine_name], heu_result_risk_14[5])
            min_negative_stocks_risk_21[vaccine_name] = min(min_negative_stocks_risk_21[vaccine_name], heu_result_risk_21[5])

            # Calculate the heuristic result
            heuristic_result[vaccine_name].append( heu_result[0] )
            heuristic_result_z[vaccine_name].append( heu_result[3] )
            penality_heuristic[vaccine_name].append( heu_result[1] )
            utilization_T_heuristic[vaccine_name].append( heu_result[2] )

            heuristic_risk_result[vaccine_name].append( heu_result_risk[0] )
            penality_heuristic_risk[vaccine_name].append( heu_result_risk[1] )
            heuristic_risk_result_z[vaccine_name].append( heu_result_risk[6] )
            heuristic_risk_negative_arrival[vaccine_name].append( heu_result_risk[2] )
            heuristic_risk_count_negative[vaccine_name].append( heu_result_risk[3] )
            utilization_T_heuristic_risk[vaccine_name].append( heu_result_risk[4] )

            heuristic_risk_result_7[vaccine_name].append( heu_result_risk_7[0] )
            heuristic_risk_result_7_z[vaccine_name].append( heu_result_risk_7[6] )
            penality_heuristic_risk_7[vaccine_name].append( heu_result_risk_7[1] )
            heuristic_risk_negative_arrival_7[vaccine_name].append( heu_result_risk_7[2] )
            heuristic_risk_count_negative_7[vaccine_name].append( heu_result_risk_7[3] )
            utilization_T_heuristic_risk_7[vaccine_name].append( heu_result_risk_7[4] )

            heuristic_risk_result_14[vaccine_name].append( heu_result_risk_14[0] )
            heuristic_risk_result_14_z[vaccine_name].append( heu_result_risk_14[6] )
            penality_heuristic_risk_14[vaccine_name].append( heu_result_risk_14[1] )
            heuristic_risk_negative_arrival_14[vaccine_name].append( heu_result_risk_14[2] )
            heuristic_risk_count_negative_14[vaccine_name].append( heu_result_risk_14[3] )
            utilization_T_heuristic_risk_14[vaccine_name].append( heu_result_risk_14[4] )

            heuristic_risk_result_21[vaccine_name].append( heu_result_risk_21[0] )
            heuristic_risk_result_21_z[vaccine_name].append( heu_result_risk_21[6] )
            penality_heuristic_risk_21[vaccine_name].append( heu_result_risk_21[1] )
            heuristic_risk_negative_arrival_21[vaccine_name].append( heu_result_risk_21[2] )
            heuristic_risk_count_negative_21[vaccine_name].append( heu_result_risk_21[3] )
            utilization_T_heuristic_risk_21[vaccine_name].append( heu_result_risk_21[4] )

            # Write the solution to the single file (create direcitory if not exists)
            output_dir = Path(CSV_OUTPUT_FOLDER + "/capacity_" + vaccine_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(CSV_OUTPUT_FOLDER + "/capacity_" + vaccine_name + "/solution_" + file_list[instance_number], sep =';' , decimal=",")

    instances = np.arange(1, len(optimal_result["1 c"][0]) + 1 - (1000 - NUMBER_OF_ELEMENT), 1).tolist()

    # optimize_test_capacity_multiple_vaccines_robust(180, DELTA, capacity["1 c"],robust_optimization)

    # Write the solution to the summary file
    df = pd.DataFrame(instances, columns= ['instance'])
    df_summary = pd.DataFrame(instances, columns= ['instance'])
    capacity_list = []

    avg_optimal_value = {}
    avg_optimal_value_z = {}
    avg_stocks_optimal = {}
    avg_utilization_T = {}

    for alpha_element in ALPHA_LIST:
        avg_optimal_value[alpha_element] = []
        avg_optimal_value_z[alpha_element] = []
        avg_stocks_optimal[alpha_element] = []
        avg_utilization_T[alpha_element] = []

    avg_heuristic_value = []
    avg_heuristic_risk_value = []
    avg_heuristic_risk_value_7 = []
    avg_heuristic_risk_value_14 = []
    avg_heuristic_risk_value_21 = []

    avg_heuristic_value_z = []
    avg_heuristic_risk_value_z = []
    avg_heuristic_risk_value_7_z = []
    avg_heuristic_risk_value_14_z = []
    avg_heuristic_risk_value_21_z = []

    avg_result_difference = []
    avg_result_difference_risk = []
    avg_result_difference_risk_7 = []
    avg_result_difference_risk_14 = []
    avg_result_difference_risk_21 = []

    avg_stocks_heuristic = []
    avg_stocks_risk = []
    avg_stocks_risk_7 = []
    avg_stocks_risk_14 = []
    avg_stocks_risk_21 = []

    avg_stocks_difference = []
    avg_stocks_difference_risk = []
    avg_stocks_difference_risk_7 = []
    avg_stocks_difference_risk_14 = []
    avg_stocks_difference_risk_21 = []

    avg_utilization_T_heuristic = []
    avg_utilization_T_heuristic_risk = []
    avg_utilization_T_heuristic_risk_7 = []
    avg_utilization_T_heuristic_risk_14 = []
    avg_utilization_T_heuristic_risk_21 = []

    avg_heuristic_risk_negative_arrival = []
    avg_heuristic_risk_negative_arrival_7 = []
    avg_heuristic_risk_negative_arrival_14 = []
    avg_heuristic_risk_negative_arrival_21 =[]

    avg_heuristic_risk_count_negative = []
    avg_heuristic_risk_count_negative_7 = []
    avg_heuristic_risk_count_negative_14 = []
    avg_heuristic_risk_count_negative_21 = []

    total_min_negative_stocks_risk = []
    total_min_negative_stocks_risk_7 = []
    total_min_negative_stocks_risk_14 = []
    total_min_negative_stocks_risk_21 = []

    for k in optimal_result:
        for alpha_element in ALPHA_LIST:
            df[k + " - optimal solution - " + str(alpha_element)] = optimal_result[k][alpha_element]
            df[k + " - optimal solution remain stocks - " + str(alpha_element)] = penality_optimal_result[k][alpha_element]

        df[k + " - conservative heuristic solution"] = heuristic_result[k]
        df[k + " - conservative heuristic remain stocks"] = penality_heuristic[k]

        df[k + " - heuristic q-1 risk solution"] = heuristic_risk_result[k]
        df[k + " - heuristic q-1 risk remain stocks"] = penality_heuristic_risk[k]

        df[k + " - heuristic q-7 risk solution"] = heuristic_risk_result_7[k]
        df[k + " - heuristic q-7risk remain stocks"] = penality_heuristic_risk_7[k]

        df[k + " - heuristic q14 risk solution"] = heuristic_risk_result_14[k]
        df[k + " - heuristic q-14 risk remain stocks"] = penality_heuristic_risk_14[k]

        df[k + " - heuristic q-21 risk solution"] = heuristic_risk_result_21[k]
        df[k + " - heuristic q-21 risk remain stocks"] = penality_heuristic_risk_21[k]
        
        capacity_list.append(k)

        for alpha_element in ALPHA_LIST:
            avg_optimal_value[alpha_element].append(round( mean(optimal_result[k][alpha_element] ), 4))
            avg_optimal_value_z[alpha_element].append(round( mean(optimal_result_z[k][alpha_element] ), 4))
            avg_stocks_optimal[alpha_element].append(round( mean(penality_optimal_result[k][alpha_element] ), 4))
            avg_utilization_T[alpha_element].append( round( mean(utilization_T[k][alpha_element] ), 4) )

        avg_heuristic_value.append(round( mean(heuristic_result[k] ), 4))
        avg_heuristic_risk_value.append(round( mean(heuristic_risk_result[k] ), 4))
        avg_heuristic_risk_value_7.append(round( mean(heuristic_risk_result_7[k] ), 4))
        avg_heuristic_risk_value_14.append(round( mean(heuristic_risk_result_14[k] ), 4))
        avg_heuristic_risk_value_21.append(round( mean(heuristic_risk_result_21[k] ), 4))

        avg_heuristic_value_z.append(round( mean(heuristic_result_z[k] ), 2))
        avg_heuristic_risk_value_z.append(round( mean(heuristic_risk_result_z[k] ), 2))
        avg_heuristic_risk_value_7_z.append(round( mean(heuristic_risk_result_7_z[k] ), 2))
        avg_heuristic_risk_value_14_z.append(round( mean(heuristic_risk_result_14_z[k] ), 2))
        avg_heuristic_risk_value_21_z.append(round( mean(heuristic_risk_result_21_z[k] ), 2))

        avg_stocks_heuristic.append(round( mean(penality_heuristic[k] ), 2))
        avg_stocks_risk.append(round( mean(penality_heuristic_risk[k] ), 2))
        avg_stocks_risk_7.append(round( mean(penality_heuristic_risk_7[k] ), 2))
        avg_stocks_risk_14.append(round( mean(penality_heuristic_risk_14[k] ), 2))
        avg_stocks_risk_21.append(round( mean(penality_heuristic_risk_21[k] ), 2))

        avg_utilization_T_heuristic.append( round( mean(utilization_T_heuristic[k] ), 5) )
        avg_utilization_T_heuristic_risk.append( round( mean(utilization_T_heuristic_risk[k] ), 5) )
        avg_utilization_T_heuristic_risk_7.append( round( mean(utilization_T_heuristic_risk_7[k] ), 5) )
        avg_utilization_T_heuristic_risk_14.append( round( mean(utilization_T_heuristic_risk_14[k] ), 5) )
        avg_utilization_T_heuristic_risk_21.append( round( mean(utilization_T_heuristic_risk_21[k] ), 5) )

        avg_heuristic_risk_count_negative.append(round (mean(heuristic_risk_count_negative[k]), 4))
        avg_heuristic_risk_count_negative_7.append(round (mean(heuristic_risk_count_negative_7[k]), 4))
        avg_heuristic_risk_count_negative_14.append(round (mean(heuristic_risk_count_negative_14[k]), 4))
        avg_heuristic_risk_count_negative_21.append(round (mean(heuristic_risk_count_negative_21[k]), 4))

        avg_heuristic_risk_negative_arrival.append(round (mean(heuristic_risk_negative_arrival[k]), 5))
        avg_heuristic_risk_negative_arrival_7.append(round (mean(heuristic_risk_negative_arrival_7[k]), 5))
        avg_heuristic_risk_negative_arrival_14.append(round (mean(heuristic_risk_negative_arrival_14[k]), 5))
        avg_heuristic_risk_negative_arrival_21.append(round (mean(heuristic_risk_negative_arrival_21[k]), 5))

        total_min_negative_stocks_risk.append(min_negative_stocks_risk[k])
        total_min_negative_stocks_risk_7.append(min_negative_stocks_risk_7[k])
        total_min_negative_stocks_risk_14.append(min_negative_stocks_risk_14[k])
        total_min_negative_stocks_risk_21.append(min_negative_stocks_risk_21[k])

    df.to_csv("result_summary_instances.csv", index=0)

    df = pd.DataFrame(capacity_list, columns= ['Capacity'])

    for alpha_element in ALPHA_LIST:
        df['LP model - ' + str(alpha_element)] = avg_optimal_value[alpha_element]

    for alpha_element in ALPHA_LIST:
        df['LP model - ' + str(alpha_element) + ' - Z value'] = avg_optimal_value_z[alpha_element]
    
    df['Conservative heuristic'] = avg_heuristic_value
    df['Heuristic 1-days-ahead risk'] = avg_heuristic_risk_value
    df['Heuristic 7-days-ahead risk'] = avg_heuristic_risk_value_7
    df['Heuristic 14-days-ahead risk'] = avg_heuristic_risk_value_14
    df['Heuristic 21-days-ahead risk'] = avg_heuristic_risk_value_21

    df['Conservative heuristic - Z'] = avg_heuristic_value_z
    df['Heuristic 1-days-ahead risk - Z'] = avg_heuristic_risk_value_z
    df['Heuristic 7-days-ahead risk - Z'] = avg_heuristic_risk_value_7_z
    df['Heuristic 14-days-ahead risk - Z'] = avg_heuristic_risk_value_14_z
    df['Heuristic 21-days-ahead risk - Z'] = avg_heuristic_risk_value_21_z

    for alpha_element in ALPHA_LIST:
        df['LP model stocks - ' + str(alpha_element)] = avg_stocks_optimal[alpha_element]
    
    df['Conservative heuristic stocks'] = avg_stocks_heuristic
    df['Heuristic q-1 risk stocks'] = avg_stocks_risk
    df['Heuristic q-7 risk stocks'] = avg_stocks_risk_7
    df['Heuristic q-14 risk stocks'] = avg_stocks_risk_14
    df['Heuristic q-21 risk stocks'] = avg_stocks_risk_21

    df['Heuristic 1-days-ahead negative on arrival'] = avg_heuristic_risk_negative_arrival
    df['Heuristic 7-days-ahead negative on arrival'] = avg_heuristic_risk_negative_arrival_7
    df['Heuristic 14-days-ahead negative on arrival'] = avg_heuristic_risk_negative_arrival_14
    df['Heuristic 21-days-ahead negative on arrival'] = avg_heuristic_risk_negative_arrival_21

    df['Heuristic 1-days-ahead negative days'] = avg_heuristic_risk_count_negative
    df['Heuristic 7-days-ahead negative days'] = avg_heuristic_risk_count_negative_7
    df['Heuristic 14-days-ahead negative days'] = avg_heuristic_risk_count_negative_14
    df['Heuristic 21-days-ahead negative days'] = avg_heuristic_risk_count_negative_21
    
    for alpha_element in ALPHA_LIST:
        df['LP model utilization - ' + str(alpha_element)] = avg_utilization_T[alpha_element]

    df['Heuristic utilization'] = avg_utilization_T_heuristic
    df['Heuristic 1-days-ahead utilization'] = avg_utilization_T_heuristic_risk
    df['Heuristic 7-days-ahead utilization'] = avg_utilization_T_heuristic_risk_7
    df['Heuristic 14-days-ahead utilization'] = avg_utilization_T_heuristic_risk_14
    df['Heuristic 21-days-ahead utilization'] = avg_utilization_T_heuristic_risk_21

    df['Heuristic 1-days-ahead max negative stocks'] = total_min_negative_stocks_risk
    df['Heuristic 7-days-ahead max negative stocks'] = total_min_negative_stocks_risk_7
    df['Heuristic 14-days-ahead max negative stocks'] = total_min_negative_stocks_risk_14
    df['Heuristic 21-days-ahead max negative stocks'] = total_min_negative_stocks_risk_21

    if LAST_DAY_VACCINES:
        df.to_csv("result_summary_zero_stocks.csv", index = 0, sep =';', decimal=",")
    else:
        df.to_csv("result_summary.csv", index = 0)