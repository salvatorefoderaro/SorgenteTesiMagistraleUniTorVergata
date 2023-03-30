library(countreg)

# Numero di file da generare
num_files <- 1000

# Percorso di output
output_path <- "../zip_arrival"

# Creazione della directory di output (se non esiste già)
dir.create(output_path, showWarnings = FALSE)

# Ciclo di generazione dei file
for (x in 1:num_files) {
  
  # Generazione dei dati casuali e salvataggio su file CSV
  data <- rzipois(180, 10000000, 0.85)
  df <- data.frame(data)
  colnames(df) <- c("ndosi")
  filename <- paste0(output_path, "zip_", x, ".csv")
  write.csv(df, filename)
  
  # Barra di avanzamento
  cat(sprintf("Generazione file %d di %d\n", x, num_files))
}