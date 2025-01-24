# Trade-in-East-Africa
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from math import exp
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import networkx as nx

#Layer 1
# File Path
file_path = '/Users/moneerayassien/PycharmProjects/analysis/.venv/lib/MastersheetV3.xlsx'

# Load Data
markets = pd.read_excel(file_path, sheet_name='MARKETS')
main = pd.read_excel(file_path, sheet_name='MAIN')
borders = pd.read_excel(file_path, sheet_name='BORDERS')

# Standardize Column Names
markets = markets.rename(columns={'MARKETID2': 'MARKETID', 'X': 'Longitude', 'Y': 'Latitude'})
main = main.rename(columns={'MARKETI D2': 'MARKETID', 'route_Length(meter)': 'route_length'})
borders = borders.rename(columns={'ID': 'BorderID', 'Border name': 'Border_Name', 'X': 'Longitude', 'Y': 'Latitude'})

# Convert MARKETS data to GeoDataFrame
markets_gdf = gpd.GeoDataFrame(
    markets,
    geometry=gpd.points_from_xy(markets['Longitude'], markets['Latitude']),
    crs="EPSG:4326"
)

# Convert BORDERS data to GeoDataFrame
borders_gdf = gpd.GeoDataFrame(
    borders,
    geometry=gpd.points_from_xy(borders['Longitude'], borders['Latitude']),
    crs="EPSG:4326"
)

# Adjust Buffers Based on Market Type
markets_gdf['Buffer_distance'] = markets_gdf['Type'].apply(
    lambda x: 1500 if x.lower() == 'urban' else 3000
)

# Calculate Buffers
markets_gdf = markets_gdf.to_crs(epsg=3857)  # Project to meters for buffer calculation
markets_gdf['Buffer'] = markets_gdf.geometry.buffer(markets_gdf['Buffer_distance'])
markets_gdf = markets_gdf.to_crs(epsg=4326)  # Reproject back to WGS84

# Merge route length data from MAIN sheet with MARKETS
merged = pd.merge(main, markets_gdf, on='MARKETID', how='left')

# Convert route lengths to kilometers
merged['route_length_km'] = merged['route_length'] / 1000

# Decay Function and MPI Calculation
def calculate_mpi(group, decay_param):
    group[f'MPI_{decay_param}'] = group.apply(
        lambda row: exp(-decay_param * row['route_length_km']) * row['POP2020'], axis=1
    )
    return group

# Calculate MPI for each decay parameter
for j in [0.02, 0.03, 0.05]:
    merged = calculate_mpi(merged, j)

# Correct Normalization: Apply Min-Max Normalization before Aggregation
for j in [0.02, 0.03, 0.05]:
    col = f'MPI_{j}'
    merged[f'Norm_MPI_{j}'] = (merged[col] - merged[col].min()) / (merged[col].max() - merged[col].min())

# Aggregate MPI Scores AFTER Normalization
mpi_market_results = merged.groupby('MARKETID')[[f'Norm_MPI_{j}' for j in [0.02, 0.03, 0.05]]].mean().reset_index()
mpi_border_results = merged.groupby('BorderID')[[f'Norm_MPI_{j}' for j in [0.02, 0.03, 0.05]]].mean().reset_index()

# Merge MPI Results with Borders Data
mpi_border_results = pd.merge(
    mpi_border_results,
    borders[['BorderID', 'Border_Name', 'Longitude', 'Latitude']],
    on='BorderID',
    how='left'
)

# Remove rows with missing Border_Name
mpi_border_results = mpi_border_results.dropna(subset=['Border_Name'])

# Directory for Saving Results
output_dir = '/Users/moneerayassien/PycharmProjects/analysis/.venv/18Jan_results'
os.makedirs(output_dir, exist_ok=True)

# Save Results
mpi_market_results.to_csv(os.path.join(output_dir, 'mpi_market_results.csv'), index=False)
mpi_border_results.to_csv(os.path.join(output_dir, 'mpi_border_results.csv'), index=False)

# Visualization: MPI by Border Post (Bar Chart)
plt.figure(figsize=(12, 8))
mpi_border_results.set_index('Border_Name')[[f'Norm_MPI_{j}' for j in [0.02, 0.03, 0.05]]].plot(kind='bar')
plt.title('Market Potential Index (MPI) by Border Post')
plt.ylabel('Market Potential Index (MPI)')
plt.xlabel('Border Name')
plt.legend(title='Decay Parameter')
plt.ylim(0, 1)  # Ensure y-axis is limited between 0 and 1
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'MPI_bar_chart.png'))
plt.close()

# Visualization: MPI Heatmap
for j in [0.02, 0.03, 0.05]:
    plt.figure(figsize=(10, 6))
    pivot_table = mpi_border_results.set_index('Border_Name')[[f'Norm_MPI_{j}']]
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": f"Market Potential Index (Decay {j})"},
        vmin=0, vmax=1  # Ensure heatmap scale is 0 to 1
    )
    plt.title(f'Market Potential Index (MPI) for Decay {j}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'MPI_heatmap_decay_{j}.png'))
    plt.close()

print(f"Results and visualizations saved in: {output_dir}")


##############Layer 2 ################################
# Directory for Layer 2 Results
layer2_output_dir = '/Users/moneerayassien/PycharmProjects/analysis/.venv/final_results/layer2'
os.makedirs(layer2_output_dir, exist_ok=True)

# Load Aligned Data
aligned_file_path = '/Users/moneerayassien/PycharmProjects/analysis/.venv/final_results'
markets_filtered = pd.read_csv(os.path.join(aligned_file_path, 'mpi_market_results.csv'))
borders_filtered = pd.read_csv(os.path.join(aligned_file_path, 'mpi_border_results.csv'))

# Load Conflict Data from the MAIN Dataset
main_conflict = pd.read_excel(file_path, sheet_name='MAIN')
main_conflict = main_conflict.rename(columns={
    'conflict event_Count': 'event_count',
    'route_Length(meter)': 'route_length'
})

# Verify necessary columns
if 'event_count' not in main_conflict.columns or 'fatalities' not in main_conflict.columns:
    raise KeyError("The required columns 'event_count' and/or 'fatalities' are missing from the MAIN dataset.")

# Calculate the Weight (CI Factors)
main_conflict['weight'] = main_conflict['fatalities'] + 0.5 * main_conflict['event_count']

# Temporal Conflict Exposure Index (CI) Calculation
ci_results = main_conflict.groupby(['year', 'BorderID']).apply(
    lambda df: np.sum(df['weight'] * (1 / (df['route_length'] / 1000)))
).reset_index(name='CI')

# Merge CI Results with Border Data
ci_results = pd.merge(
    ci_results,
    borders_filtered[['BorderID', 'Border_Name', 'Longitude', 'Latitude']],
    on='BorderID',
    how='left'
)

# Save CI Results
ci_results.to_csv(os.path.join(layer2_output_dir, 'ci_results.csv'), index=False)

# Ensure 'ci_results' has unique combinations of 'Border_Name' and 'year'
ci_results_agg = ci_results.groupby(['Border_Name', 'year'], as_index=False).agg({'CI': 'sum'})

# Visualization: Temporal Trends in CI
plt.figure(figsize=(12, 8))
sns.lineplot(data=ci_results_agg, x='year', y='CI', hue='Border_Name', marker="o")
plt.title('Temporal Trends in Conflict Exposure Index (CI)')
plt.ylabel('CI Value')
plt.xlabel('Year')
plt.legend(title='Border Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(layer2_output_dir, 'ci_temporal_trends.png'))
plt.close()

# Visualization: Heatmap of CI
ci_pivot = ci_results_agg.pivot(index='Border_Name', columns='year', values='CI')
plt.figure(figsize=(12, 8))
sns.heatmap(ci_pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "CI Value"})
plt.title('Conflict Exposure Index (CI) by Border and Year')
plt.ylabel('Border Name')
plt.xlabel('Year')
plt.tight_layout()
plt.savefig(os.path.join(layer2_output_dir, 'ci_heatmap.png'))
plt.close()

# Print confirmation of successful visualization
print(f"Temporal analysis visualizations saved in: {layer2_output_dir}")


#Layer 3
import geopandas as gpd
data = gpd.read_file("/Users/moneerayassien/PycharmProjects/analysis/.venv/lib/EA202.geojson")

# Define bounding box for Kenya and Uganda
latitude_min, latitude_max = -5, 5
longitude_min, longitude_max = 29, 42

# Filter the data based on the bounding box
kenya_uganda_data = data[
    (data.geometry.x >= longitude_min) & (data.geometry.x <= longitude_max) &
    (data.geometry.y >= latitude_min) & (data.geometry.y <= latitude_max)
]

# Display the filtered data
kenya_uganda_data.head()

# Ensure ethnic data uses the same CRS as other GeoDataFrames
kenya_uganda_data = kenya_uganda_data.to_crs(markets_gdf.crs)

# Save the filtered ethnic data for future use
ethnic_output_path = '/Users/moneerayassien/PycharmProjects/analysis/.venv/lib/kenya_uganda_ethnic.geojson'
kenya_uganda_data.to_file(ethnic_output_path, driver="GeoJSON")
print(f"Filtered ethnic data saved to: {ethnic_output_path}")

ethnic_output_path = '/Users/moneerayassien/PycharmProjects/analysis/.venv/lib/kenya_uganda_ethnic.geojson'
kenya_uganda_data.to_file(ethnic_output_path, driver="GeoJSON")
# Spatial join: Ethnic data with MPI border results
ethnic_mpi_overlay = gpd.sjoin(kenya_uganda_data, borders_gdf, how="inner", predicate="intersects")

# Merge MPI values to the ethnic data
ethnic_mpi_overlay = pd.merge(
    ethnic_mpi_overlay,
    mpi_border_results[['BorderID', 'Norm_MPI_0.02', 'Norm_MPI_0.03', 'Norm_MPI_0.05']],
    on='BorderID',
    how='left'
)

# Save the result
ethnic_mpi_output_path = '/Users/moneerayassien/PycharmProjects/analysis/.venv/lib/ethnic_mpi_overlay.geojson'
ethnic_mpi_overlay.to_file(ethnic_mpi_output_path, driver="GeoJSON")
print(f"Ethnic-MPI overlay saved to: {ethnic_mpi_output_path}")

# Ensure CI results have a geometry column
ci_results_gdf = gpd.GeoDataFrame(
    ci_results,
    geometry=gpd.points_from_xy(ci_results['Longitude'], ci_results['Latitude']),
    crs="EPSG:4326"  # Set CRS to WGS84
)
# Aggregate MPI values by ethnic group
mpi_summary = ethnic_mpi_overlay.groupby('name')[['Norm_MPI_0.02', 'Norm_MPI_0.03', 'Norm_MPI_0.05']].mean()

# Save MPI summary to a CSV file
mpi_summary_path = '/Users/moneerayassien/PycharmProjects/analysis/.venv/lib/mpi_summary.csv'
mpi_summary.to_csv(mpi_summary_path)
print(f"MPI summary saved to: {mpi_summary_path}")

# Define output directory
output_dir = '/Users/moneerayassien/PycharmProjects/analysis/.venv/18Jan_results'

# Aggregate MPI values by ethnic group
mpi_summary = ethnic_mpi_overlay.groupby('name')[['Norm_MPI_0.02', 'Norm_MPI_0.03', 'Norm_MPI_0.05']].mean()

# Save MPI summary to a CSV file
mpi_summary_path = os.path.join(output_dir, 'mpi_summary.csv')
mpi_summary.to_csv(mpi_summary_path)
print(f"MPI summary saved to: {mpi_summary_path}")
