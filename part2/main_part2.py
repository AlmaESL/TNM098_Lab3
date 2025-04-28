import pandas as pd
import plotly.graph_objects as go
from preprocessing import load_and_process
from filter import get_relevant

# Filepath
filepath = 'TNM098-MC3-2011.csv'

# Load and process the data
df = load_and_process(filepath)

# Parse date column
df['Date'] = pd.to_datetime(df['Date'])

# Get relevant (filtered) data
relevant_df = get_relevant(df)

# --- Group counts per day ---
# Count total number of reports per day (unfiltered)
unfiltered_counts = df['Date'].dt.date.value_counts().sort_index()

# Count number of relevant reports per day (filtered)
filtered_counts = relevant_df['Date'].dt.date.value_counts().sort_index()

# --- Create the Figure ---
fig = go.Figure()

# Add unfiltered data (background, gray bars)
fig.add_trace(go.Bar(
    x=unfiltered_counts.index,
    y=unfiltered_counts.values,
    name='All Reports',
    marker_color='gray',
    opacity=0.6
))

# Add filtered data (foreground, blue bars)
fig.add_trace(go.Bar(
    x=filtered_counts.index,
    y=filtered_counts.values,
    name='Relevant Reports',
    marker_color='blue',
    opacity=0.8
))

# Update layout
fig.update_layout(
    title='News Report Frequency (Unfiltered and Filtered)',
    barmode='overlay',  # Overlapping bars
    xaxis_title='Date',
    yaxis_title='Count',
    xaxis_tickformat='%b %d',
    legend_title='Legend'
)

fig.show()
