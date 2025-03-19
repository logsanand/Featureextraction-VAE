import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from osgeo import gdal, gdal_array
from dbfread import DBF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score

# Define file paths (Modify these paths accordingly before running)
data_dir = "path/to/data"  # Change this to the directory containing your files
p_2018 = os.path.join(data_dir, "pinkegat_nov18_file_rem.dbf")
z_2018 = os.path.join(data_dir, "zoutkamperlaag_aug18_file_all.dbf")
p_2019 = os.path.join(data_dir, "pinkegat_feb19_file_all.dbf")
z_2019 = os.path.join(data_dir, "zoutkamperlaag_aug19_file_all.dbf")
p_2020 = os.path.join(data_dir, "pinkegat_apr20_file_all.dbf")
z_2020 = os.path.join(data_dir, "zoutkamperlaag_apr20_file_all_with.dbf")

# Load DBF files into Pandas DataFrames
def load_dbf(file_path):
    return pd.DataFrame(iter(DBF(file_path)))

df_p2018, df_z2018 = load_dbf(p_2018), load_dbf(z_2018)
df_p2019, df_z2019 = load_dbf(p_2019), load_dbf(z_2019)
df_p2020, df_z2020 = load_dbf(p_2020), load_dbf(z_2020)

# Ensure column consistency
def standardize_columns(df):
    df.columns = range(df.shape[1])
    return df

df_p2018, df_z2018 = standardize_columns(df_p2018), standardize_columns(df_z2018)
df_p2019, df_z2019 = standardize_columns(df_p2019), standardize_columns(df_z2019)
df_p2020, df_z2020 = standardize_columns(df_p2020), standardize_columns(df_z2020)

# Sort data by relevant columns
def sort_df(df):
    return df.sort_values([df.columns[2], df.columns[1]])

df_p2018, df_z2018 = sort_df(df_p2018), sort_df(df_z2018)
df_p2019, df_z2019 = sort_df(df_p2019), sort_df(df_z2019)
df_p2020, df_z2020 = sort_df(df_p2020), sort_df(df_z2020)

# Prepare training data
fea = pd.concat([df_p2019, df_z2019], ignore_index=True, sort=False)
fea = fea.iloc[:, 5:].abs()
fea = fea[fea.iloc[:, 0] != 0]  # Remove zero values

# Define features and target variable
X = fea.iloc[:, 1:]
y = fea.iloc[:, 0]

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=800, random_state=42, max_depth=20, 
                                 max_features='sqrt', min_samples_split=10, min_samples_leaf=2, bootstrap=False)
rf_model.fit(X, y)
print("Model R² Score:", r2_score(y, rf_model.predict(X)))

# Load raster images (Modify file paths)
image_path = "path/to/original_image.tif"
feature_path = "path/to/encoder_image.tif"
shapefile_path = "path/to/study_area.shp"

dataset = gdal.Open(image_path)
geo_transform = dataset.GetGeoTransform()
image_array = gdal_array.LoadFile(image_path)
feature_array = gdal_array.LoadFile(feature_path)

# Generate coordinate grid
xmin, ymax = geo_transform[0], geo_transform[3]
xres, yres = geo_transform[1], geo_transform[5]
W, H = dataset.RasterXSize, dataset.RasterYSize
xmax = xmin + (W * xres)
ymin = ymax + (H * yres)
xy_grid = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]

# Prepare data for prediction
stacked_data = np.concatenate((image_array, feature_array), axis=0)
df_grid = pd.DataFrame(stacked_data.reshape([stacked_data.shape[0], -1]).T)
df_grid['lon'], df_grid['lat'] = xy_grid[0].T.flatten(), xy_grid[1].T.flatten()
df_grid = df_grid[df_grid.iloc[:, 0] >= 0]

# Make predictions
df_grid['pred'] = rf_model.predict(df_grid.iloc[:, :-2])

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df_grid['lon'], df_grid['lat'])]
gdf_predictions = gpd.GeoDataFrame(df_grid[['lon', 'lat', 'pred']], geometry=geometry)

# Load study area shapefile and clip predictions
study_area = gpd.read_file(shapefile_path)
gdf_clipped = gpd.clip(gdf_predictions, study_area).set_crs('epsg:32631')

# Save results to shapefile
output_shapefile = "path/to/output_predictions.shp"
gdf_clipped.to_file(output_shapefile, driver='ESRI Shapefile')
print("Predictions saved to:", output_shapefile)
