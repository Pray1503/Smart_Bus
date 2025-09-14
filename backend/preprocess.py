# # #!/usr/bin/env python3
# # """
# # Smart Bus Management System - Data Preprocessing Script
# # Cleans and preprocesses the generated bus data for Prophet forecasting
# # """

# # import pandas as pd
# # import numpy as np
# # import json
# # from datetime import datetime, timedelta
# # from pathlib import Path

# # class BusDataPreprocessor:
# #     """Preprocess bus ridership and GPS data for forecasting"""
    
# #     def __init__(self, data_dir='/app/backend/data'):
# #         self.data_dir = Path(data_dir)
# #         self.processed_data = {}
        
# #     def load_raw_data(self):
# #         """Load raw generated data"""
# #         print("Loading raw data files...")
        
# #         # Load GPS logs
# #         gps_file = self.data_dir / 'gps_logs.csv'
# #         if gps_file.exists():
# #             self.gps_data = pd.read_csv(gps_file)
# #             self.gps_data['timestamp'] = pd.to_datetime(self.gps_data['timestamp'])
# #             print(f"Loaded {len(self.gps_data)} GPS records")
        
# #         # Load ridership data
# #         ridership_file = self.data_dir / 'ridership.json'
# #         if ridership_file.exists():
# #             with open(ridership_file, 'r') as f:
# #                 ridership_list = json.load(f)
# #             self.ridership_data = pd.DataFrame(ridership_list)
# #             self.ridership_data['timestamp'] = pd.to_datetime(self.ridership_data['timestamp'])
# #             print(f"Loaded {len(self.ridership_data)} ridership records")
        
# #         # Load routes master
# #         routes_file = self.data_dir / 'routes_master.csv'
# #         if routes_file.exists():
# #             self.routes_data = pd.read_csv(routes_file)
# #             print(f"Loaded {len(self.routes_data)} route records")
    
# #     def clean_gps_data(self):
# #         """Clean and validate GPS tracking data"""
# #         print("Cleaning GPS data...")
        
# #         if not hasattr(self, 'gps_data'):
# #             return
        
# #         initial_count = len(self.gps_data)
        
# #         # Remove invalid coordinates (basic validation for Bangalore area)
# #         bangalore_bounds = {
# #             'lat_min': 12.5, 'lat_max': 13.5,
# #             'lng_min': 77.0, 'lng_max': 78.0
# #         }
        
# #         valid_coords = (
# #             (self.gps_data['lat'] >= bangalore_bounds['lat_min']) &
# #             (self.gps_data['lat'] <= bangalore_bounds['lat_max']) &
# #             (self.gps_data['lng'] >= bangalore_bounds['lng_min']) &
# #             (self.gps_data['lng'] <= bangalore_bounds['lng_max'])
# #         )
        
# #         self.gps_data = self.gps_data[valid_coords]
        
# #         # Remove invalid speeds (negative or unreasonably high)
# #         self.gps_data = self.gps_data[
# #             (self.gps_data['speed'] >= 0) & 
# #             (self.gps_data['speed'] <= 80)  # Max 80 km/h for city buses
# #         ]
        
# #         # Sort by timestamp and route
# #         self.gps_data = self.gps_data.sort_values(['route_id', 'bus_id', 'timestamp'])
        
# #         # Add calculated features
# #         self.gps_data['hour'] = self.gps_data['timestamp'].dt.hour
# #         self.gps_data['day_of_week'] = self.gps_data['timestamp'].dt.dayofweek
# #         self.gps_data['is_weekend'] = self.gps_data['day_of_week'] >= 5
# #         self.gps_data['is_rush_hour'] = self.gps_data['hour'].isin([7, 8, 9, 17, 18, 19])
        
# #         cleaned_count = len(self.gps_data)
# #         print(f"GPS data cleaned: {initial_count} -> {cleaned_count} records")
        
# #         self.processed_data['gps_clean'] = self.gps_data
    
# #     def clean_ridership_data(self):
# #         """Clean and validate ridership data"""
# #         print("Cleaning ridership data...")
        
# #         if not hasattr(self, 'ridership_data'):
# #             return
        
# #         initial_count = len(self.ridership_data)
        
# #         # Remove negative ridership
# #         self.ridership_data = self.ridership_data[self.ridership_data['ridership'] >= 0]
        
# #         # Cap occupancy at 100%
# #         self.ridership_data['occupancy_percent'] = self.ridership_data['occupancy_percent'].clip(0, 100)
        
# #         # Fill missing weather factors
# #         self.ridership_data['weather_factor'] = self.ridership_data['weather_factor'].fillna(1.0)
        
# #         # Sort by timestamp
# #         self.ridership_data = self.ridership_data.sort_values(['route_id', 'timestamp'])
        
# #         # Create rolling averages for smoothing
# #         self.ridership_data['ridership_rolling_3h'] = (
# #             self.ridership_data.groupby('route_id')['ridership']
# #             .rolling(window=3, min_periods=1)
# #             .mean()
# #             .reset_index(0, drop=True)
# #         )
        
# #         # Remove outliers using IQR method
# #         for route_id in self.ridership_data['route_id'].unique():
# #             route_mask = self.ridership_data['route_id'] == route_id
# #             route_data = self.ridership_data[route_mask]['ridership']
            
# #             Q1 = route_data.quantile(0.25)
# #             Q3 = route_data.quantile(0.75)
# #             IQR = Q3 - Q1
            
# #             outlier_mask = (
# #                 (route_data < (Q1 - 1.5 * IQR)) |
# #                 (route_data > (Q3 + 1.5 * IQR))
# #             )
            
# #             # Replace outliers with rolling average
# #             self.ridership_data.loc[route_mask & outlier_mask, 'ridership'] = (
# #                 self.ridership_data.loc[route_mask & outlier_mask, 'ridership_rolling_3h']
# #             )
        
# #         cleaned_count = len(self.ridership_data)
# #         print(f"Ridership data cleaned: {initial_count} -> {cleaned_count} records")
        
# #         self.processed_data['ridership_clean'] = self.ridership_data
    
# #     def create_prophet_format_data(self):
# #         """Create Prophet-compatible time series data"""
# #         print("Creating Prophet format data...")
        
# #         if 'ridership_clean' not in self.processed_data:
# #             return
        
# #         ridership_data = self.processed_data['ridership_clean']
# #         prophet_datasets = {}
        
# #         for route_id in ridership_data['route_id'].unique():
# #             route_data = ridership_data[ridership_data['route_id'] == route_id].copy()
            
# #             # Create Prophet format (ds, y columns)
# #             prophet_df = pd.DataFrame({
# #                 'ds': route_data['timestamp'],
# #                 'y': route_data['ridership']
# #             })
            
# #             # Add external regressors
# #             prophet_df['is_weekend'] = route_data['is_weekend'].astype(int)
# #             prophet_df['is_rush_hour'] = route_data['is_rush_hour'].astype(int)
# #             prophet_df['weather_factor'] = route_data['weather_factor']
# #             prophet_df['hour'] = route_data['hour']
# #             prophet_df['day_of_week'] = route_data['day_of_week']
            
# #             # Add temperature simulation (since we don't have real weather data)
# #             prophet_df['temperature'] = self._simulate_temperature(prophet_df['ds'])
# #             prophet_df['temp_extreme'] = (
# #                 (prophet_df['temperature'] < 15) | 
# #                 (prophet_df['temperature'] > 35)
# #             ).astype(int)
            
# #             # Add precipitation simulation
# #             prophet_df['precipitation'] = self._simulate_precipitation(prophet_df['ds'])
# #             prophet_df['heavy_rain'] = (prophet_df['precipitation'] > 10).astype(int)
            
# #             # Sort by timestamp
# #             prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
            
# #             prophet_datasets[f'route_{route_id}'] = prophet_df
            
# #         self.processed_data['prophet_data'] = prophet_datasets
# #         print(f"Created Prophet datasets for {len(prophet_datasets)} routes")
    
# #     def _simulate_temperature(self, timestamps):
# #         """Simulate temperature data based on time patterns"""
# #         temps = []
# #         for ts in timestamps:
# #             # Base temperature with seasonal and daily variations
# #             day_of_year = ts.timetuple().tm_yday
# #             hour = ts.hour
            
# #             # Seasonal pattern (Bangalore climate)
# #             seasonal_temp = 25 + 5 * np.sin(2 * np.pi * day_of_year / 365)
            
# #             # Daily pattern
# #             daily_variation = 3 * np.sin(2 * np.pi * (hour - 6) / 24)
            
# #             # Random variation
# #             random_variation = np.random.normal(0, 2)
            
# #             temperature = seasonal_temp + daily_variation + random_variation
# #             temps.append(max(10, min(40, temperature)))  # Reasonable bounds
        
# #         return temps
    
# #     def _simulate_precipitation(self, timestamps):
# #         """Simulate precipitation data"""
# #         precip = []
# #         for ts in timestamps:
# #             # Higher chance of rain during monsoon months (June-September)
# #             month = ts.month
# #             if 6 <= month <= 9:  # Monsoon season
# #                 rain_chance = 0.3
# #                 base_rain = 5
# #             else:
# #                 rain_chance = 0.1
# #                 base_rain = 1
            
# #             if np.random.random() < rain_chance:
# #                 precipitation = np.random.exponential(base_rain)
# #             else:
# #                 precipitation = 0
            
# #             precip.append(min(50, precipitation))  # Cap at 50mm
        
# #         return precip
    
# #     def aggregate_data_for_optimization(self):
# #         """Create aggregated data for route optimization"""
# #         print("Creating optimization datasets...")
        
# #         if 'ridership_clean' not in self.processed_data:
# #             return
        
# #         ridership_data = self.processed_data['ridership_clean']
        
# #         # Hourly aggregation by route
# #         hourly_agg = ridership_data.groupby(['route_id', 'hour']).agg({
# #             'ridership': ['mean', 'std', 'count'],
# #             'occupancy_percent': 'mean',
# #             'weather_factor': 'mean'
# #         }).round(2)
        
# #         hourly_agg.columns = ['_'.join(col).strip() for col in hourly_agg.columns]
# #         hourly_agg = hourly_agg.reset_index()
        
# #         # Daily aggregation by route
# #         ridership_data['date'] = ridership_data['timestamp'].dt.date
# #         daily_agg = ridership_data.groupby(['route_id', 'date']).agg({
# #             'ridership': ['sum', 'mean', 'max'],
# #             'occupancy_percent': 'mean',
# #             'is_weekend': 'first'
# #         }).round(2)
        
# #         daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns]
# #         daily_agg = daily_agg.reset_index()
        
# #         self.processed_data['hourly_aggregation'] = hourly_agg
# #         self.processed_data['daily_aggregation'] = daily_agg
        
# #         print(f"Created aggregated datasets: hourly ({len(hourly_agg)} records), daily ({len(daily_agg)} records)")
    
# #     def save_processed_data(self):
# #         """Save all processed datasets"""
# #         print("Saving processed data...")
        
# #         output_dir = self.data_dir
        
# #         # Save cleaned datasets
# #         if 'gps_clean' in self.processed_data:
# #             self.processed_data['gps_clean'].to_csv(output_dir / 'gps_logs_clean.csv', index=False)
            
# #         if 'ridership_clean' in self.processed_data:
# #             self.processed_data['ridership_clean'].to_csv(output_dir / 'ridership_clean.csv', index=False)
        
# #         # Save Prophet format data
# #         if 'prophet_data' in self.processed_data:
# #             for route_name, route_df in self.processed_data['prophet_data'].items():
# #                 route_df.to_csv(output_dir / f'prophet_{route_name}.csv', index=False)
        
# #         # Save aggregated data
# #         if 'hourly_aggregation' in self.processed_data:
# #             self.processed_data['hourly_aggregation'].to_csv(output_dir / 'hourly_aggregation.csv', index=False)
            
# #         if 'daily_aggregation' in self.processed_data:
# #             self.processed_data['daily_aggregation'].to_csv(output_dir / 'daily_aggregation.csv', index=False)
        
# #         # Create master clean dataset combining all routes
# #         if 'prophet_data' in self.processed_data:
# #             all_routes_data = []
# #             for route_name, route_df in self.processed_data['prophet_data'].items():
# #                 route_id = route_name.split('_')[1]
# #                 route_df_copy = route_df.copy()
# #                 route_df_copy['route_id'] = int(route_id)
# #                 all_routes_data.append(route_df_copy)
            
# #             master_df = pd.concat(all_routes_data, ignore_index=True)
# #             master_df.to_csv(output_dir / 'clean_data.csv', index=False)
# #             print(f"Saved master clean dataset with {len(master_df)} records")
        
# #         # Generate processing summary
# #         summary = {
# #             'processing_date': datetime.now().isoformat(),
# #             'datasets_processed': list(self.processed_data.keys()),
# #             'record_counts': {
# #                 name: len(df) for name, df in self.processed_data.items() 
# #                 if isinstance(df, pd.DataFrame)
# #             },
# #             'data_quality_metrics': self._calculate_quality_metrics()
# #         }
        
# #         with open(output_dir / 'preprocessing_summary.json', 'w') as f:
# #             json.dump(summary, f, indent=2)
        
# #         print("Processing completed successfully!")
# #         print(f"Summary: {summary}")
    
# #     def _calculate_quality_metrics(self):
# #         """Calculate data quality metrics"""
# #         metrics = {}
        
# #         if 'ridership_clean' in self.processed_data:
# #             ridership_data = self.processed_data['ridership_clean']
            
# #             metrics['ridership'] = {
# #                 'completeness': (1 - ridership_data['ridership'].isna().sum() / len(ridership_data)) * 100,
# #                 'zero_values_pct': (ridership_data['ridership'] == 0).sum() / len(ridership_data) * 100,
# #                 'avg_ridership': ridership_data['ridership'].mean(),
# #                 'std_ridership': ridership_data['ridership'].std()
# #             }
        
# #         if 'gps_clean' in self.processed_data:
# #             gps_data = self.processed_data['gps_clean']
            
# #             metrics['gps'] = {
# #                 'completeness': (1 - gps_data[['lat', 'lng']].isna().sum().sum() / (2 * len(gps_data))) * 100,
# #                 'unique_buses': gps_data['bus_id'].nunique(),
# #                 'unique_routes': gps_data['route_id'].nunique(),
# #                 'avg_speed': gps_data['speed'].mean()
# #             }
        
# #         return metrics
    
# #     def run_preprocessing(self):
# #         """Run complete preprocessing pipeline"""
# #         print("Starting data preprocessing pipeline...")
        
# #         self.load_raw_data()
# #         self.clean_gps_data()
# #         self.clean_ridership_data()
# #         self.create_prophet_format_data()
# #         self.aggregate_data_for_optimization()
# #         self.save_processed_data()
        
# #         print("\nData preprocessing pipeline completed successfully!")

# # if __name__ == "__main__":
# #     processor = BusDataPreprocessor()
# #     processor.run_preprocessing()

# #!/usr/bin/env python3
# """
# Smart Bus Management System - Prophet Forecasting Model
# Implements demand prediction using Facebook Prophet with external regressors
# """

# import pandas as pd
# import numpy as np
# from prophet import Prophet
# from prophet.diagnostics import cross_validation, performance_metrics
# import joblib
# import json
# from datetime import datetime, timedelta
# from pathlib import Path
# import warnings
# import logging

# # Suppress Prophet warnings
# warnings.filterwarnings('ignore')
# logging.getLogger('prophet').setLevel(logging.WARNING)

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Define root directory and data path
# ROOT_DIR = Path(__file__).parent
# DATA_PATH = ROOT_DIR / 'data' / 'clean_data.csv'

# class BusRidershipForecaster:
#     """Prophet-based forecasting model for bus ridership demand"""
    
#     def __init__(self):
#         self.models = {}
#         self.model_metadata = {}
#         self.regressors = [
#             'is_weekend', 'is_rush_hour', 'weather_factor', 
#             'temp_extreme', 'heavy_rain', 'hour'
#         ]
        
#     def prepare_data_for_training(self, df, route_id=None):
#         """Prepare data for Prophet training"""
#         logger.info(f"Preparing data for route_id={route_id}")
#         if route_id:
#             df = df[df['route_id'] == route_id].copy()
        
#         # Ensure required columns exist
#         required_cols = ['ds', 'y']
#         for col in required_cols:
#             if col not in df.columns:
#                 raise ValueError(f"Missing required column: {col}")
        
#         # Add defaults for missing regressors and ensure correct types
#         for col in self.regressors:
#             if col not in df.columns:
#                 if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
#                     df[col] = 0  # Default to 0 (integer)
#                 elif col == 'weather_factor':
#                     df[col] = 1.0
#                 elif col == 'hour':
#                     df[col] = df['ds'].dt.hour.astype(float)
#             else:
#                 if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
#                     df[col] = df[col].astype(int)  # Convert to integer (0/1)
#                 else:
#                     df[col] = df[col].astype(float)  # Ensure numeric
        
#         # Select only necessary columns to exclude extras like strings
#         columns_to_keep = required_cols + self.regressors
#         df = df[columns_to_keep]
        
#         # Convert to datetime
#         df['ds'] = pd.to_datetime(df['ds'])
        
#         # Sort by timestamp
#         df = df.sort_values('ds').reset_index(drop=True)
        
#         # Remove duplicates
#         df = df.drop_duplicates(subset=['ds'])
        
#         # Ensure non-negative values
#         df['y'] = df['y'].clip(lower=0).astype(float)
        
#         # Log data info for debugging
#         logger.info(f"Prepared data: {len(df)} rows, columns: {df.columns.tolist()}")
#         logger.info(f"Data types: {df.dtypes.to_dict()}")
#         logger.info(f"Sample data:\n{df.head().to_dict()}")
        
#         return df
    
#     def create_prophet_model(self, route_id):
#         """Create and configure Prophet model for specific route"""
#         route_params = {
#             500: {  # High-frequency city route
#                 'changepoint_prior_scale': 0.1,
#                 'seasonality_prior_scale': 15.0,
#                 'daily_seasonality': True
#             },
#             501: {  # IT corridor route
#                 'changepoint_prior_scale': 0.05,
#                 'seasonality_prior_scale': 20.0,
#                 'daily_seasonality': True
#             },
#             502: {  # Tech hub route
#                 'changepoint_prior_scale': 0.08,
#                 'seasonality_prior_scale': 25.0,
#                 'daily_seasonality': True
#             },
#             503: {  # Residential route
#                 'changepoint_prior_scale': 0.05,
#                 'seasonality_prior_scale': 10.0,
#                 'daily_seasonality': False
#             },
#             504: {  # Commercial route
#                 'changepoint_prior_scale': 0.07,
#                 'seasonality_prior_scale': 18.0,
#                 'daily_seasonality': True
#             }
#         }
        
#         params = route_params.get(route_id, {
#             'changepoint_prior_scale': 0.05,
#             'seasonality_prior_scale': 15.0,
#             'daily_seasonality': True
#         })
        
#         model = Prophet(
#             changepoint_prior_scale=params['changepoint_prior_scale'],
#             seasonality_prior_scale=params['seasonality_prior_scale'],
#             holidays_prior_scale=10.0,
#             daily_seasonality=params['daily_seasonality'],
#             weekly_seasonality=True,
#             yearly_seasonality=False,  # Not enough data for yearly
#             interval_width=0.8
#         )
        
#         # Add external regressors
#         regressor_configs = {
#             'is_weekend': {'mode': 'additive', 'prior_scale': 2.0},
#             'is_rush_hour': {'mode': 'multiplicative', 'prior_scale': 1.5},
#             'weather_factor': {'mode': 'multiplicative', 'prior_scale': 1.0},
#             'temp_extreme': {'mode': 'additive', 'prior_scale': 1.0},
#             'heavy_rain': {'mode': 'additive', 'prior_scale': 1.5},
#             'hour': {'mode': 'additive', 'prior_scale': 0.5}
#         }
        
#         for regressor, config in regressor_configs.items():
#             model.add_regressor(
#                 regressor,
#                 mode=config['mode'],
#                 prior_scale=config['prior_scale']
#             )
        
#         return model
    
#     def train_route_model(self, df, route_id):
#         """Train Prophet model for specific route"""
#         print(f"Training model for Route {route_id}...")
        
#         try:
#             # Prepare data
#             train_df = self.prepare_data_for_training(df, route_id)
            
#             if len(train_df) < 24:  # Need at least 24 hours of data
#                 print(f"Insufficient data for Route {route_id}: {len(train_df)} records")
#                 return None
            
#             # Create and configure model
#             model = self.create_prophet_model(route_id)
            
#             # Ensure regressors are integer (0/1)
#             train_df['is_rush_hour'] = train_df['is_rush_hour'].astype(int)
            
#             # Log sample data for debugging
#             logger.info(f"Training data for Route {route_id}:\n{train_df.head().to_dict()}")
            
#             # Train model
#             start_time = datetime.now()
#             model.fit(train_df)
#             training_time = (datetime.now() - start_time).total_seconds()
            
#             # Store model and metadata
#             self.models[route_id] = model
#             self.model_metadata[route_id] = {
#                 'training_date': datetime.now().isoformat(),
#                 'training_records': len(train_df),
#                 'training_time_seconds': training_time,
#                 'data_start': train_df['ds'].min().isoformat(),
#                 'data_end': train_df['ds'].max().isoformat()
#             }
            
#             print(f"Route {route_id} model trained successfully ({training_time:.2f}s)")
#             return model
        
#         except Exception as e:
#             print(f"Failed to train model for Route {route_id}: {str(e)}")
#             return None
    
#     def train_all_models(self, data_path=DATA_PATH):
#         """Train models for all routes"""
#         print(f"Loading clean data from {data_path}...")
        
#         try:
#             logger.info(f"Attempting to read file: {data_path}")
#             df = pd.read_csv(data_path)
#             df['ds'] = pd.to_datetime(df['ds'])
#         except FileNotFoundError:
#             logging.error(f"Data file not found at {data_path}")
#             return False
#         except Exception as e:
#             logging.error(f"Error loading data: {str(e)}")
#             return False
        
#         routes = df['route_id'].unique()
#         print(f"Training models for {len(routes)} routes...")
        
#         success_count = 0
#         for route_id in routes:
#             try:
#                 model = self.train_route_model(df, route_id)
#                 if model:
#                     success_count += 1
#             except Exception as e:
#                 print(f"Failed to train model for Route {route_id}: {str(e)}")
        
#         print(f"Successfully trained {success_count}/{len(routes)} models")
#         return success_count > 0
    
#     def generate_forecast(self, route_id, hours_ahead=24, external_regressors=None):
#         """Generate forecast for specific route"""
#         if route_id not in self.models:
#             raise ValueError(f"No trained model found for Route {route_id}")
        
#         model = self.models[route_id]
        
#         # Create future dataframe
#         last_date = datetime.now().replace(minute=0, second=0, microsecond=0)
#         future_dates = pd.date_range(
#             start=last_date + timedelta(hours=1),
#             periods=hours_ahead,
#             freq='H'
#         )
        
#         future_df = pd.DataFrame({
#             'ds': future_dates
#         })
        
#         # Add time-based regressors
#         future_df['hour'] = future_df['ds'].dt.hour
#         future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
#         future_df['is_rush_hour'] = future_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
#         # Add weather regressors (use defaults if not provided)
#         if external_regressors is not None:
#             for col in ['weather_factor', 'temp_extreme', 'heavy_rain']:
#                 if col in external_regressors:
#                     future_df[col] = external_regressors[col][:len(future_df)]
#                 else:
#                     # Default values
#                     if col == 'weather_factor':
#                         future_df[col] = 1.0
#                     else:
#                         future_df[col] = 0
#         else:
#             future_df['weather_factor'] = 1.0
#             future_df['temp_extreme'] = 0
#             future_df['heavy_rain'] = 0
        
#         # Generate forecast
#         forecast = model.predict(future_df)
        
#         # Ensure non-negative predictions
#         forecast['yhat'] = forecast['yhat'].clip(lower=0)
#         forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
#         forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
#         return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
#     def evaluate_model_performance(self, route_id, data_path=DATA_PATH):
#         """Evaluate model performance using cross-validation"""
#         if route_id not in self.models:
#             print(f"No model found for Route {route_id}")
#             return None
        
#         print(f"Evaluating model performance for Route {route_id}...")
        
#         try:
#             logger.info(f"Attempting to read file for evaluation: {data_path}")
#             df = pd.read_csv(data_path)
#             df['ds'] = pd.to_datetime(df['ds'])
#         except FileNotFoundError:
#             logging.error(f"Data file not found at {data_path}")
#             return None
#         except Exception as e:
#             logging.error(f"Error loading data: {str(e)}")
#             return None
        
#         route_df = self.prepare_data_for_training(df, route_id)
        
#         if len(route_df) < 72:  # Need at least 3 days for CV
#             print(f"Insufficient data for cross-validation: {len(route_df)} records")
#             return None
        
#         try:
#             model = self.models[route_id]
            
#             # Perform cross-validation
#             cv_results = cross_validation(
#                 model,
#                 initial='48 hours',
#                 period='12 hours',
#                 horizon='24 hours'
#             )
            
#             # Calculate performance metrics
#             metrics = performance_metrics(cv_results)
            
#             performance = {
#                 'mae': metrics['mae'].mean(),
#                 'rmse': metrics['rmse'].mean(),
#                 'mape': metrics['mape'].mean(),
#                 'coverage': ((cv_results['y'] >= cv_results['yhat_lower']) & 
#                              (cv_results['y'] <= cv_results['yhat_upper'])).mean()
#             }
            
#             print(f"Route {route_id} Performance - MAE: {performance['mae']:.2f}, RMSE: {performance['rmse']:.2f}")
#             return performance
            
#         except Exception as e:
#             print(f"Cross-validation failed for Route {route_id}: {str(e)}")
#             return None
    
#     def save_models(self, model_dir=ROOT_DIR / 'models'):
#         """Save trained models to disk"""
#         model_path = Path(model_dir)
#         model_path.mkdir(exist_ok=True)
        
#         for route_id, model in self.models.items():
#             model_file = model_path / f'prophet_route_{route_id}.pkl'
            
#             # Save model with metadata
#             model_data = {
#                 'model': model,
#                 'metadata': self.model_metadata.get(route_id, {}),
#                 'regressors': self.regressors
#             }
            
#             joblib.dump(model_data, model_file)
#             print(f"Saved model for Route {route_id}")
        
#         # Save overall metadata
#         metadata_file = model_path / 'models_metadata.json'
#         with open(metadata_file, 'w') as f:
#             json.dump(self.model_metadata, f, indent=2)
        
#         print(f"All models saved to {model_dir}")
    
#     def load_models(self, model_dir=ROOT_DIR / 'models'):
#         """Load trained models from disk"""
#         model_path = Path(model_dir)
        
#         if not model_path.exists():
#             print(f"Model directory {model_dir} does not exist")
#             return False
        
#         model_files = list(model_path.glob('prophet_route_*.pkl'))
        
#         if not model_files:
#             print("No model files found")
#             return False
        
#         loaded_count = 0
#         for model_file in model_files:
#             try:
#                 # Extract route ID from filename
#                 route_id = int(model_file.stem.split('_')[-1])
                
#                 # Load model
#                 model_data = joblib.load(model_file)
#                 self.models[route_id] = model_data['model']
#                 self.model_metadata[route_id] = model_data.get('metadata', {})
                
#                 loaded_count += 1
#                 print(f"Loaded model for Route {route_id}")
                
#             except Exception as e:
#                 print(f"Failed to load {model_file}: {str(e)}")
        
#         print(f"Loaded {loaded_count} models successfully")
#         return loaded_count > 0

# def main():
#     """Main function to train and save models"""
#     print("Starting Prophet model training...")
    
#     forecaster = BusRidershipForecaster()
    
#     # Train models
#     success = forecaster.train_all_models()
    
#     if success:
#         # Save models
#         forecaster.save_models()
        
#         # Evaluate performance for one route as example
#         performance = forecaster.evaluate_model_performance(500)
#         if performance:
#             print(f"Sample performance metrics: {performance}")
        
#         print("Model training completed successfully!")
#     else:
#         print("Model training failed!")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Smart Bus Management System - Prophet Forecasting Model
Implements demand prediction using Facebook Prophet with external regressors
"""

# import pandas as pd
# import numpy as np
# from prophet import Prophet
# from prophet.diagnostics import cross_validation, performance_metrics
# import joblib
# import json
# from datetime import datetime, timedelta
# from pathlib import Path
# import warnings
# import logging

# # Suppress Prophet warnings
# warnings.filterwarnings('ignore')
# logging.getLogger('prophet').setLevel(logging.WARNING)

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Define root directory and data path
# ROOT_DIR = Path(__file__).parent
# DATA_PATH = ROOT_DIR / 'data' / 'clean_data.csv'

# class BusRidershipForecaster:
#     """Prophet-based forecasting model for bus ridership demand"""
    
#     def __init__(self):
#         self.models = {}
#         self.model_metadata = {}
#         self.regressors = [
#             'is_weekend', 'is_rush_hour', 'weather_factor', 
#             'temp_extreme', 'heavy_rain', 'hour'
#         ]
        
#     def prepare_data_for_training(self, df, route_id=None):
#         """Prepare data for Prophet training"""
#         logger.info(f"Preparing data for route_id={route_id}")
#         if route_id:
#             df = df[df['route_id'] == route_id].copy()
        
#         # Ensure required columns exist
#         required_cols = ['ds', 'y']
#         for col in required_cols:
#             if col not in df.columns:
#                 raise ValueError(f"Missing required column: {col}")
        
#         # Add defaults for missing regressors and ensure correct types
#         for col in self.regressors:
#             if col not in df.columns:
#                 if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
#                     df[col] = 0  # Default to 0 (integer)
#                 elif col == 'weather_factor':
#                     df[col] = 1.0
#                 elif col == 'hour':
#                     df[col] = df['ds'].dt.hour.astype(float)
#             else:
#                 if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
#                     df[col] = df[col].astype(int)  # Convert to integer (0/1)
#                 else:
#                     df[col] = df[col].astype(float)  # Ensure numeric
        
#         # Select only necessary columns to exclude extras like strings
#         columns_to_keep = required_cols + self.regressors
#         df = df[columns_to_keep]
        
#         # Convert to datetime
#         df['ds'] = pd.to_datetime(df['ds'])
        
#         # Sort by timestamp
#         df = df.sort_values('ds').reset_index(drop=True)
        
#         # Remove duplicates
#         df = df.drop_duplicates(subset=['ds'])
        
#         # Ensure non-negative values
#         df['y'] = df['y'].clip(lower=0).astype(float)
        
#         # Log data info for debugging
#         logger.info(f"Prepared data: {len(df)} rows, columns: {df.columns.tolist()}")
#         logger.info(f"Data types: {df.dtypes.to_dict()}")
#         logger.info(f"Sample data:\n{df.head().to_dict()}")
        
#         return df
    
#     def create_prophet_model(self, route_id):
#         """Create and configure Prophet model for specific route"""
#         route_params = {
#             500: {  # High-frequency city route
#                 'changepoint_prior_scale': 0.1,
#                 'seasonality_prior_scale': 15.0,
#                 'daily_seasonality': True
#             },
#             501: {  # IT corridor route
#                 'changepoint_prior_scale': 0.05,
#                 'seasonality_prior_scale': 20.0,
#                 'daily_seasonality': True
#             },
#             502: {  # Tech hub route
#                 'changepoint_prior_scale': 0.08,
#                 'seasonality_prior_scale': 25.0,
#                 'daily_seasonality': True
#             },
#             503: {  # Residential route
#                 'changepoint_prior_scale': 0.05,
#                 'seasonality_prior_scale': 10.0,
#                 'daily_seasonality': False
#             },
#             504: {  # Commercial route
#                 'changepoint_prior_scale': 0.07,
#                 'seasonality_prior_scale': 18.0,
#                 'daily_seasonality': True
#             }
#         }
        
#         params = route_params.get(route_id, {
#             'changepoint_prior_scale': 0.05,
#             'seasonality_prior_scale': 15.0,
#             'daily_seasonality': True
#         })
        
#         model = Prophet(
#             changepoint_prior_scale=params['changepoint_prior_scale'],
#             seasonality_prior_scale=params['seasonality_prior_scale'],
#             holidays_prior_scale=10.0,
#             daily_seasonality=params['daily_seasonality'],
#             weekly_seasonality=True,
#             yearly_seasonality=False,  # Not enough data for yearly
#             interval_width=0.8
#         )
        
#         # Add external regressors
#         regressor_configs = {
#             'is_weekend': {'mode': 'additive', 'prior_scale': 2.0},
#             'is_rush_hour': {'mode': 'multiplicative', 'prior_scale': 1.5},
#             'weather_factor': {'mode': 'multiplicative', 'prior_scale': 1.0},
#             'temp_extreme': {'mode': 'additive', 'prior_scale': 1.0},
#             'heavy_rain': {'mode': 'additive', 'prior_scale': 1.5},
#             'hour': {'mode': 'additive', 'prior_scale': 0.5}
#         }
        
#         for regressor, config in regressor_configs.items():
#             model.add_regressor(
#                 regressor,
#                 mode=config['mode'],
#                 prior_scale=config['prior_scale']
#             )
        
#         return model
    
#     def train_route_model(self, df, route_id):
#         """Train Prophet model for specific route"""
#         print(f"Training model for Route {route_id}...")
        
#         try:
#             # Convert route_id to Python int to avoid numpy.int64
#             route_id = int(route_id)
            
#             # Prepare data
#             train_df = self.prepare_data_for_training(df, route_id)
            
#             if len(train_df) < 24:  # Need at least 24 hours of data
#                 print(f"Insufficient data for Route {route_id}: {len(train_df)} records")
#                 return None
            
#             # Create and configure model
#             model = self.create_prophet_model(route_id)
            
#             # Ensure regressors are integer (0/1)
#             train_df['is_rush_hour'] = train_df['is_rush_hour'].astype(int)
            
#             # Log sample data for debugging
#             logger.info(f"Training data for Route {route_id}:\n{train_df.head().to_dict()}")
            
#             # Train model
#             start_time = datetime.now()
#             model.fit(train_df)
#             training_time = (datetime.now() - start_time).total_seconds()
            
#             # Store model and metadata
#             self.models[route_id] = model
#             self.model_metadata[route_id] = {
#                 'training_date': datetime.now().isoformat(),
#                 'training_records': len(train_df),
#                 'training_time_seconds': training_time,
#                 'data_start': train_df['ds'].min().isoformat(),
#                 'data_end': train_df['ds'].max().isoformat()
#             }
            
#             print(f"Route {route_id} model trained successfully ({training_time:.2f}s)")
#             return model
        
#         except Exception as e:
#             print(f"Failed to train model for Route {route_id}: {str(e)}")
#             return None
    
#     def train_all_models(self, data_path=DATA_PATH):
#         """Train models for all routes"""
#         print(f"Loading clean data from {data_path}...")
        
#         try:
#             logger.info(f"Attempting to read file: {data_path}")
#             df = pd.read_csv(data_path)
#             df['ds'] = pd.to_datetime(df['ds'])
#             # Convert route_id to Python int
#             df['route_id'] = df['route_id'].astype(int)
#         except FileNotFoundError:
#             logging.error(f"Data file not found at {data_path}")
#             return False
#         except Exception as e:
#             logging.error(f"Error loading data: {str(e)}")
#             return False
        
#         # Convert route IDs to Python int
#         routes = [int(route_id) for route_id in df['route_id'].unique()]
#         print(f"Training models for {len(routes)} routes...")
        
#         success_count = 0
#         for route_id in routes:
#             try:
#                 model = self.train_route_model(df, route_id)
#                 if model:
#                     success_count += 1
#             except Exception as e:
#                 print(f"Failed to train model for Route {route_id}: {str(e)}")
        
#         print(f"Successfully trained {success_count}/{len(routes)} models")
#         return success_count > 0
    
#     def generate_forecast(self, route_id, hours_ahead=24, external_regressors=None):
#         """Generate forecast for specific route"""
#         route_id = int(route_id)  # Ensure route_id is Python int
#         if route_id not in self.models:
#             raise ValueError(f"No trained model found for Route {route_id}")
        
#         model = self.models[route_id]
        
#         # Create future dataframe
#         last_date = datetime.now().replace(minute=0, second=0, microsecond=0)
#         future_dates = pd.date_range(
#             start=last_date + timedelta(hours=1),
#             periods=hours_ahead,
#             freq='H'
#         )
        
#         future_df = pd.DataFrame({
#             'ds': future_dates
#         })
        
#         # Add time-based regressors
#         future_df['hour'] = future_df['ds'].dt.hour
#         future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
#         future_df['is_rush_hour'] = future_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
#         # Add weather regressors (use defaults if not provided)
#         if external_regressors is not None:
#             for col in ['weather_factor', 'temp_extreme', 'heavy_rain']:
#                 if col in external_regressors:
#                     future_df[col] = external_regressors[col][:len(future_df)]
#                 else:
#                     # Default values
#                     if col == 'weather_factor':
#                         future_df[col] = 1.0
#                     else:
#                         future_df[col] = 0
#         else:
#             future_df['weather_factor'] = 1.0
#             future_df['temp_extreme'] = 0
#             future_df['heavy_rain'] = 0
        
#         # Generate forecast
#         forecast = model.predict(future_df)
        
#         # Ensure non-negative predictions
#         forecast['yhat'] = forecast['yhat'].clip(lower=0)
#         forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
#         forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
#         return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
#     def evaluate_model_performance(self, route_id, data_path=DATA_PATH):
#         """Evaluate model performance using cross-validation"""
#         route_id = int(route_id)  # Ensure route_id is Python int
#         if route_id not in self.models:
#             print(f"No model found for Route {route_id}")
#             return None
        
#         print(f"Evaluating model performance for Route {route_id}...")
        
#         try:
#             logger.info(f"Attempting to read file for evaluation: {data_path}")
#             df = pd.read_csv(data_path)
#             df['ds'] = pd.to_datetime(df['ds'])
#             df['route_id'] = df['route_id'].astype(int)
#         except FileNotFoundError:
#             logging.error(f"Data file not found at {data_path}")
#             return None
#         except Exception as e:
#             logging.error(f"Error loading data: {str(e)}")
#             return None
        
#         route_df = self.prepare_data_for_training(df, route_id)
        
#         if len(route_df) < 72:  # Need at least 3 days for CV
#             print(f"Insufficient data for cross-validation: {len(route_df)} records")
#             return None
        
#         try:
#             model = self.models[route_id]
            
#             # Perform cross-validation
#             cv_results = cross_validation(
#                 model,
#                 initial='48 hours',
#                 period='12 hours',
#                 horizon='24 hours'
#             )
            
#             # Calculate performance metrics
#             metrics = performance_metrics(cv_results)
            
#             performance = {
#                 'mae': metrics['mae'].mean(),
#                 'rmse': metrics['rmse'].mean(),
#                 'mape': metrics['mape'].mean(),
#                 'coverage': ((cv_results['y'] >= cv_results['yhat_lower']) & 
#                              (cv_results['y'] <= cv_results['yhat_upper'])).mean()
#             }
            
#             print(f"Route {route_id} Performance - MAE: {performance['mae']:.2f}, RMSE: {performance['rmse']:.2f}")
#             return performance
            
#         except Exception as e:
#             print(f"Cross-validation failed for Route {route_id}: {str(e)}")
#             return None
    
#     def save_models(self, model_dir=ROOT_DIR / 'models'):
#         """Save trained models to disk"""
#         model_path = Path(model_dir)
#         model_path.mkdir(exist_ok=True)
        
#         for route_id, model in self.models.items():
#             route_id = int(route_id)  # Ensure route_id is Python int
#             model_file = model_path / f'prophet_route_{route_id}.pkl'
            
#             # Save model with metadata
#             model_data = {
#                 'model': model,
#                 'metadata': self.model_metadata.get(route_id, {}),
#                 'regressors': self.regressors
#             }
            
#             joblib.dump(model_data, model_file)
#             print(f"Saved model for Route {route_id}")
        
#         # Save overall metadata
#         metadata_file = model_path / 'models_metadata.json'
#         with open(metadata_file, 'w') as f:
#             json.dump(self.model_metadata, f, indent=2)
        
#         print(f"All models saved to {model_dir}")
    
#     def load_models(self, model_dir=ROOT_DIR / 'models'):
#         """Load trained models from disk"""
#         model_path = Path(model_dir)
        
#         if not model_path.exists():
#             print(f"Model directory {model_dir} does not exist")
#             return False
        
#         model_files = list(model_path.glob('prophet_route_*.pkl'))
        
#         if not model_files:
#             print("No model files found")
#             return False
        
#         loaded_count = 0
#         for model_file in model_files:
#             try:
#                 # Extract route ID from filename
#                 route_id = int(model_file.stem.split('_')[-1])
                
#                 # Load model
#                 model_data = joblib.load(model_file)
#                 self.models[route_id] = model_data['model']
#                 self.model_metadata[route_id] = model_data.get('metadata', {})
                
#                 loaded_count += 1
#                 print(f"Loaded model for Route {route_id}")
                
#             except Exception as e:
#                 print(f"Failed to load {model_file}: {str(e)}")
        
#         print(f"Loaded {loaded_count} models successfully")
#         return loaded_count > 0

# def main():
#     """Main function to train and save models"""
#     print("Starting Prophet model training...")
    
#     forecaster = BusRidershipForecaster()
    
#     # Train models
#     success = forecaster.train_all_models()
    
#     if success:
#         # Save models
#         forecaster.save_models()
        
#         # Evaluate performance for one route as example
#         performance = forecaster.evaluate_model_performance(500)
#         if performance:
#             print(f"Sample performance metrics: {performance}")
        
#         print("Model training completed successfully!")
#     else:
#         print("Model training failed!")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Smart Bus Management System - Data Preprocessing Script
Cleans and preprocesses the generated bus data for Prophet forecasting
"""
import re
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

class BusDataPreprocessor:
    """Preprocess bus ridership and GPS data for forecasting"""
    
    def __init__(self, data_dir='/app/backend/data'):
        self.data_dir = Path(data_dir)
        self.processed_data = {}
        
    def load_raw_data(self):
        """Load raw generated data"""
        print("Loading raw data files...")
        
        # Load GPS logs
        gps_file = self.data_dir / 'gps_logs.csv'
        if gps_file.exists():
            self.gps_data = pd.read_csv(gps_file)
            self.gps_data['timestamp'] = pd.to_datetime(self.gps_data['timestamp'])
            print(f"Loaded {len(self.gps_data)} GPS records")
        
        # Load ridership data
        ridership_file = self.data_dir / 'ridership.json'
        if ridership_file.exists():
            with open(ridership_file, 'r') as f:
                ridership_list = json.load(f)
            self.ridership_data = pd.DataFrame(ridership_list)
            self.ridership_data['timestamp'] = pd.to_datetime(self.ridership_data['timestamp'])
            print(f"Loaded {len(self.ridership_data)} ridership records")
        
        # Load routes master
        routes_file = self.data_dir / 'routes_master.csv'
        if routes_file.exists():
            self.routes_data = pd.read_csv(routes_file)
            print(f"Loaded {len(self.routes_data)} route records")
    
    def clean_gps_data(self):
        """Clean and validate GPS tracking data"""
        print("Cleaning GPS data...")
        
        if not hasattr(self, 'gps_data'):
            return
        
        initial_count = len(self.gps_data)
        
        # Remove invalid coordinates (basic validation for Bangalore area)
        bangalore_bounds = {
            'lat_min': 12.5, 'lat_max': 13.5,
            'lng_min': 77.0, 'lng_max': 78.0
        }
        
        valid_coords = (
            (self.gps_data['lat'] >= bangalore_bounds['lat_min']) &
            (self.gps_data['lat'] <= bangalore_bounds['lat_max']) &
            (self.gps_data['lng'] >= bangalore_bounds['lng_min']) &
            (self.gps_data['lng'] <= bangalore_bounds['lng_max'])
        )
        
        self.gps_data = self.gps_data[valid_coords]
        
        # Remove invalid speeds (negative or unreasonably high)
        self.gps_data = self.gps_data[
            (self.gps_data['speed'] >= 0) & 
            (self.gps_data['speed'] <= 80)  # Max 80 km/h for city buses
        ]
        
        # Sort by timestamp and route
        self.gps_data = self.gps_data.sort_values(['route_id', 'bus_id', 'timestamp'])
        
        # Add calculated features
        self.gps_data['hour'] = self.gps_data['timestamp'].dt.hour
        self.gps_data['day_of_week'] = self.gps_data['timestamp'].dt.dayofweek
        self.gps_data['is_weekend'] = self.gps_data['day_of_week'] >= 5
        self.gps_data['is_rush_hour'] = self.gps_data['hour'].isin([7, 8, 9, 17, 18, 19])
        
        cleaned_count = len(self.gps_data)
        print(f"GPS data cleaned: {initial_count} -> {cleaned_count} records")
        
        self.processed_data['gps_clean'] = self.gps_data
    
    def clean_ridership_data(self):
        """Clean and validate ridership data"""
        print("Cleaning ridership data...")
        
        if not hasattr(self, 'ridership_data'):
            return
        
        initial_count = len(self.ridership_data)
        
        # Remove negative ridership
        self.ridership_data = self.ridership_data[self.ridership_data['ridership'] >= 0]
        
        # Cap occupancy at 100%
        self.ridership_data['occupancy_percent'] = self.ridership_data['occupancy_percent'].clip(0, 100)
        
        # Fill missing weather factors
        self.ridership_data['weather_factor'] = self.ridership_data['weather_factor'].fillna(1.0)
        
        # Sort by timestamp
        self.ridership_data = self.ridership_data.sort_values(['route_id', 'timestamp'])
        
        # Create rolling averages for smoothing
        self.ridership_data['ridership_rolling_3h'] = (
            self.ridership_data.groupby('route_id')['ridership']
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Remove outliers using IQR method
        for route_id in self.ridership_data['route_id'].unique():
            route_mask = self.ridership_data['route_id'] == route_id
            route_data = self.ridership_data[route_mask]['ridership']
            
            Q1 = route_data.quantile(0.25)
            Q3 = route_data.quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_mask = (
                (route_data < (Q1 - 1.5 * IQR)) |
                (route_data > (Q3 + 1.5 * IQR))
            )
            
            # Replace outliers with rolling average
            self.ridership_data.loc[route_mask & outlier_mask, 'ridership'] = (
                self.ridership_data.loc[route_mask & outlier_mask, 'ridership_rolling_3h']
            )
        
        cleaned_count = len(self.ridership_data)
        print(f"Ridership data cleaned: {initial_count} -> {cleaned_count} records")
        
        self.processed_data['ridership_clean'] = self.ridership_data
    
    def create_prophet_format_data(self):
        """Create Prophet-compatible time series data"""
        print("Creating Prophet format data...")
        
        if 'ridership_clean' not in self.processed_data:
            return
        
        ridership_data = self.processed_data['ridership_clean']
        prophet_datasets = {}
        
        for route_id in ridership_data['route_id'].unique():
            route_data = ridership_data[ridership_data['route_id'] == route_id].copy()
            
            # Create Prophet format (ds, y columns)
            prophet_df = pd.DataFrame({
                'ds': route_data['timestamp'],
                'y': route_data['ridership']
            })
            
            # Add external regressors
            prophet_df['is_weekend'] = route_data['is_weekend'].astype(int)
            prophet_df['is_rush_hour'] = route_data['is_rush_hour'].astype(int)
            prophet_df['weather_factor'] = route_data['weather_factor']
            prophet_df['hour'] = route_data['hour']
            prophet_df['day_of_week'] = route_data['day_of_week']
            
            # Add temperature simulation (since we don't have real weather data)
            prophet_df['temperature'] = self._simulate_temperature(prophet_df['ds'])
            prophet_df['temp_extreme'] = (
                (prophet_df['temperature'] < 15) | 
                (prophet_df['temperature'] > 35)
            ).astype(int)
            
            # Add precipitation simulation
            prophet_df['precipitation'] = self._simulate_precipitation(prophet_df['ds'])
            prophet_df['heavy_rain'] = (prophet_df['precipitation'] > 10).astype(int)
            
            # Sort by timestamp
            prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
            
            prophet_datasets[f'route_{route_id}'] = prophet_df
            
        self.processed_data['prophet_data'] = prophet_datasets
        print(f"Created Prophet datasets for {len(prophet_datasets)} routes")
    
    def _simulate_temperature(self, timestamps):
        """Simulate temperature data based on time patterns"""
        temps = []
        for ts in timestamps:
            # Base temperature with seasonal and daily variations
            day_of_year = ts.timetuple().tm_yday
            hour = ts.hour
            
            # Seasonal pattern (Bangalore climate)
            seasonal_temp = 25 + 5 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Daily pattern
            daily_variation = 3 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Random variation
            random_variation = np.random.normal(0, 2)
            
            temperature = seasonal_temp + daily_variation + random_variation
            temps.append(max(10, min(40, temperature)))  # Reasonable bounds
        
        return temps
    
    def _simulate_precipitation(self, timestamps):
        """Simulate precipitation data"""
        precip = []
        for ts in timestamps:
            # Higher chance of rain during monsoon months (June-September)
            month = ts.month
            if 6 <= month <= 9:  # Monsoon season
                rain_chance = 0.3
                base_rain = 5
            else:
                rain_chance = 0.1
                base_rain = 1
            
            if np.random.random() < rain_chance:
                precipitation = np.random.exponential(base_rain)
            else:
                precipitation = 0
            
            precip.append(min(50, precipitation))  # Cap at 50mm
        
        return precip
    
    def aggregate_data_for_optimization(self):
        """Create aggregated data for route optimization"""
        print("Creating optimization datasets...")
        
        if 'ridership_clean' not in self.processed_data:
            return
        
        ridership_data = self.processed_data['ridership_clean']
        
        # Hourly aggregation by route
        hourly_agg = ridership_data.groupby(['route_id', 'hour']).agg({
            'ridership': ['mean', 'std', 'count'],
            'occupancy_percent': 'mean',
            'weather_factor': 'mean'
        }).round(2)
        
        hourly_agg.columns = ['_'.join(col).strip() for col in hourly_agg.columns]
        hourly_agg = hourly_agg.reset_index()
        
        # Daily aggregation by route
        ridership_data['date'] = ridership_data['timestamp'].dt.date
        daily_agg = ridership_data.groupby(['route_id', 'date']).agg({
            'ridership': ['sum', 'mean', 'max'],
            'occupancy_percent': 'mean',
            'is_weekend': 'first'
        }).round(2)
        
        daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns]
        daily_agg = daily_agg.reset_index()
        
        self.processed_data['hourly_aggregation'] = hourly_agg
        self.processed_data['daily_aggregation'] = daily_agg
        
        print(f"Created aggregated datasets: hourly ({len(hourly_agg)} records), daily ({len(daily_agg)} records)")
    
    def save_processed_data(self):
        """Save all processed datasets"""
        print("Saving processed data...")
        
        output_dir = self.data_dir
        output_dir.mkdir(exist_ok=True)  # Ensure output directory exists
        
        # Save cleaned datasets
        if 'gps_clean' in self.processed_data:
            self.processed_data['gps_clean'].to_csv(output_dir / 'gps_logs_clean.csv', index=False)
            
        if 'ridership_clean' in self.processed_data:
            self.processed_data['ridership_clean'].to_csv(output_dir / 'ridership_clean.csv', index=False)
        
        # Save Prophet format data
        if 'prophet_data' in self.processed_data:
            for route_name, route_df in self.processed_data['prophet_data'].items():
                route_df.to_csv(output_dir / f'prophet_{route_name}.csv', index=False)
        
        # Save aggregated data
        if 'hourly_aggregation' in self.processed_data:
            self.processed_data['hourly_aggregation'].to_csv(output_dir / 'hourly_aggregation.csv', index=False)
            
        if 'daily_aggregation' in self.processed_data:
            self.processed_data['daily_aggregation'].to_csv(output_dir / 'daily_aggregation.csv', index=False)
        
        # Create master clean dataset combining all routes
        if 'prophet_data' in self.processed_data:
            all_routes_data = []
            for route_name, route_df in self.processed_data['prophet_data'].items():
                # Extract numeric route_id from route_name
                route_id_str = route_name.split('_')[1]  # e.g., '123 SH'
                match = re.search(r'\d+', route_id_str)
                if match:
                    route_id = float(match.group())
                else:
                    route_id = 0.0  # fallback if no number is found
                
                route_df_copy = route_df.copy()
                route_df_copy['route_id'] = route_id
                all_routes_data.append(route_df_copy)
            
            master_df = pd.concat(all_routes_data, ignore_index=True)
            master_df.to_csv(self.data_dir / 'clean_data.csv', index=False)
            print(f"Saved master clean dataset with {len(master_df)} records")
        
        # Generate processing summary
        summary = {
            'processing_date': datetime.now().isoformat(),
            'datasets_processed': list(self.processed_data.keys()),
            'record_counts': {
                name: len(df) for name, df in self.processed_data.items() 
                if isinstance(df, pd.DataFrame)
            },
            'data_quality_metrics': self._calculate_quality_metrics()
        }
        
        with open(output_dir / 'preprocessing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("Processing completed successfully!")
        print(f"Summary: {summary}")
    
    def _calculate_quality_metrics(self):
        """Calculate data quality metrics"""
        metrics = {}
        
        if 'ridership_clean' in self.processed_data:
            ridership_data = self.processed_data['ridership_clean']
            
            metrics['ridership'] = {
                'completeness': (1 - ridership_data['ridership'].isna().sum() / len(ridership_data)) * 100,
                'zero_values_pct': (ridership_data['ridership'] == 0).sum() / len(ridership_data) * 100,
                'avg_ridership': ridership_data['ridership'].mean(),
                'std_ridership': ridership_data['ridership'].std()
            }
        
        if 'gps_clean' in self.processed_data:
            gps_data = self.processed_data['gps_clean']
            
            metrics['gps'] = {
                'completeness': (1 - gps_data[['lat', 'lng']].isna().sum().sum() / (2 * len(gps_data))) * 100,
                'unique_buses': gps_data['bus_id'].nunique(),
                'unique_routes': gps_data['route_id'].nunique(),
                'avg_speed': gps_data['speed'].mean()
            }
        
        return metrics
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        self.load_raw_data()
        self.clean_gps_data()
        self.clean_ridership_data()
        self.create_prophet_format_data()
        self.aggregate_data_for_optimization()
        self.save_processed_data()
        
        print("\nData preprocessing pipeline completed successfully!")

if __name__ == "__main__":
    processor = BusDataPreprocessor()
    processor.run_preprocessing()