import pandas as pd

# Carica il tuo file Excel
df = pd.read_csv(r"C:\Users\leoli\OneDrive - Politecnico di Milano\primo anno mag\competition recsys\RecSysProject\recomm.csv")

# Rimuove le parentesi dalla lista e la converte in un formato separato da spazi
df['item_list'] = df['item_list'].str.strip('[]').str.replace(',', ' ')

# Unisci user_id con item_list
df['formatted'] = df['user_id'].astype(str) + ", " + df['item_list']

# Salva il nuovo file CSV
df[['user_id','item_list']].to_csv('output.csv', index=False)
