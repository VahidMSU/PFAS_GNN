import pandas as pd

path = 'results/GridSearchResults.csv'
df = pd.read_csv(path).dropna()

# Check if 'gnn_model' column exists
if 'gnn_model' not in df.columns:
    raise KeyError("'gnn_model' column not found in the CSV file")

print("total number of models: ", len(df))

# Sort first by the lowest val_loss, test_loss, and train_loss
df = df.sort_values(by=['val_loss','test_loss',  'train_loss'], ascending=True)

# Now get the best model for each gnn_model based on lowest test_loss and then lowest val_loss
df = df.reset_index()
df = df.groupby('gnn_model').first().drop(columns=['index'])
## sort now
df = df.sort_values(by=['test_loss','val_loss',  'train_loss'], ascending=True)
### loss with 4 decimal points
df = df.round(4)
df.to_csv('results/best_models.csv')


