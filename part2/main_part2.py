import pandas as pd
import plotly.express as px
from preprocessing import load_and_process

# Filepath
filepath = 'C:/Users/almal/Desktop/termin8/TNM098/lab 3/TNM098_Lab3/TNM098-MC3-2011.csv'

# Load and process the data
df = load_and_process(filepath)

# Look at top data entries 
# print(df[['Content', 'processed_content']].head())

# Temporal Distribution (Unfiltered)
df['Date'] = pd.to_datetime(df['Date'])

fig = px.histogram(df, x='Date', nbins=50, title='News Report Frequency')
fig.show()
