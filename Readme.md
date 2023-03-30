# Sorgente Tesi Magistrale - Foderaro Salvatore

Codice sorgente per l'implementazione del lavoro di Tesi Magistrale: "Modelli per la pianificazione di una campagna vaccinale di massa con incertezza nelle forniture".

## Struttura del repository

- **Python:**
  - **pl_euristica_offline:** implementazione in Python per il confronto tra il modello di programmazione lineare e l'algoritmo "off-line" euristico
  - **pl_online:** implementazione in Python per il confronto tra il modello di programmazione lineare e gli algoritmi "on-line" *conservativo* e *q-days-ahead*
  - **../csv_solution:** cartella di output dei file *.csv* generati dallo script Python contenente la soluzione per ogni singola istanza
- **R:** implementazione in R della simulazione del processo degli arrivi
- **zip_arrival:** cartella di output dei file *.csv* generati dallo script in R
- **runme.sh:** script per l'esecuzione, in ordine, dello script R e dei due script Python