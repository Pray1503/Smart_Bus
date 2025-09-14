# # #!/usr/bin/env python3
# # """
# # Smart Bus Management System - Prophet Forecasting Model
# # Implements demand prediction using Facebook Prophet with external regressors
# # """

# # import pandas as pd
# # import numpy as np
# # from prophet import Prophet
# # from prophet.diagnostics import cross_validation, performance_metrics
# # import joblib
# # import json
# # from datetime import datetime, timedelta
# # from pathlib import Path
# # import warnings
# # import logging

# # # Suppress Prophet warnings
# # warnings.filterwarnings('ignore')
# # logging.getLogger('prophet').setLevel(logging.WARNING)

# # class BusRidershipForecaster:
# #     """Prophet-based forecasting model for bus ridership demand"""
    
# #     def __init__(self):
# #         self.models = {}
# #         self.model_metadata = {}
# #         self.regressors = [
# #             'is_weekend', 'is_rush_hour', 'weather_factor', 
# #             'temp_extreme', 'heavy_rain', 'hour'
# #         ]
        
# #     def prepare_data_for_training(self, df, route_id=None):
# #         """Prepare data for Prophet training"""
# #         if route_id:
# #             df = df[df['route_id'] == route_id].copy()
        
# #         # Ensure required columns exist
# #         required_cols = ['ds', 'y']
# #         for col in required_cols:
# #             if col not in df.columns:
# #                 raise ValueError(f"Missing required column: {col}")
        
# #         # Convert to datetime
# #         df['ds'] = pd.to_datetime(df['ds'])
        
# #         # Sort by timestamp
# #         df = df.sort_values('ds').reset_index(drop=True)
        
# #         # Remove duplicates
# #         df = df.drop_duplicates(subset=['ds'])
        
# #         # Ensure non-negative values
# #         df['y'] = df['y'].clip(lower=0)
        
# #         return df
    
# #     def create_prophet_model(self, route_id):
# #         """Create and configure Prophet model for specific route"""
        
# #         # Route-specific parameters based on characteristics
# #         route_params = {
# #             500: {  # High-frequency city route
# #                 'changepoint_prior_scale': 0.1,
# #                 'seasonality_prior_scale': 15.0,
# #                 'daily_seasonality': True
# #             },
# #             501: {  # IT corridor route
# #                 'changepoint_prior_scale': 0.05,
# #                 'seasonality_prior_scale': 20.0,
# #                 'daily_seasonality': True
# #             },
# #             502: {  # Tech hub route
# #                 'changepoint_prior_scale': 0.08,
# #                 'seasonality_prior_scale': 25.0,
# #                 'daily_seasonality': True
# #             },
# #             503: {  # Residential route
# #                 'changepoint_prior_scale': 0.05,
# #                 'seasonality_prior_scale': 10.0,
# #                 'daily_seasonality': False
# #             },
# #             504: {  # Commercial route
# #                 'changepoint_prior_scale': 0.07,
# #                 'seasonality_prior_scale': 18.0,
# #                 'daily_seasonality': True
# #             }
# #         }
        
# #         params = route_params.get(route_id, {
# #             'changepoint_prior_scale': 0.05,
# #             'seasonality_prior_scale': 15.0,
# #             'daily_seasonality': True
# #         })
        
# #         model = Prophet(
# #             changepoint_prior_scale=params['changepoint_prior_scale'],
# #             seasonality_prior_scale=params['seasonality_prior_scale'],
# #             holidays_prior_scale=10.0,
# #             daily_seasonality=params['daily_seasonality'],
# #             weekly_seasonality=True,
# #             yearly_seasonality=False,  # Not enough data for yearly
# #             interval_width=0.8
# #         )
        
# #         # Add custom seasonalities for bus patterns
# #         model.add_seasonality(
# #             name='rush_hour_pattern',
# #             period=24,
# #             fourier_order=8,
# #             condition_name='is_rush_hour'
# #         )
        
# #         # Add external regressors
# #         regressor_configs = {
# #             'is_weekend': {'mode': 'additive', 'prior_scale': 2.0},
# #             'is_rush_hour': {'mode': 'multiplicative', 'prior_scale': 1.5},
# #             'weather_factor': {'mode': 'multiplicative', 'prior_scale': 1.0},
# #             'temp_extreme': {'mode': 'additive', 'prior_scale': 1.0},
# #             'heavy_rain': {'mode': 'additive', 'prior_scale': 1.5},
# #             'hour': {'mode': 'additive', 'prior_scale': 0.5}
# #         }
        
# #         for regressor, config in regressor_configs.items():
# #             model.add_regressor(
# #                 regressor,
# #                 mode=config['mode'],
# #                 prior_scale=config['prior_scale']
# #             )
        
# #         return model
    
# #     def train_route_model(self, df, route_id):
# #         """Train Prophet model for specific route"""
# #         print(f"Training model for Route {route_id}...")
        
# #         # Prepare data
# #         train_df = self.prepare_data_for_training(df, route_id)
        
# #         if len(train_df) < 24:  # Need at least 24 hours of data
# #             print(f"Insufficient data for Route {route_id}: {len(train_df)} records")
# #             return None
        
# #         # Create and configure model
# #         model = self.create_prophet_model(route_id)
        
# #         # Add conditional columns for seasonalities
# #         train_df['is_rush_hour'] = train_df.get('is_rush_hour', 0)
        
# #         # Train model
# #         start_time = datetime.now()
# #         model.fit(train_df)
# #         training_time = (datetime.now() - start_time).total_seconds()
        
# #         # Store model and metadata
# #         self.models[route_id] = model
# #         self.model_metadata[route_id] = {
# #             'training_date': datetime.now().isoformat(),
# #             'training_records': len(train_df),
# #             'training_time_seconds': training_time,
# #             'data_start': train_df['ds'].min().isoformat(),
# #             'data_end': train_df['ds'].max().isoformat()
# #         }
        
# #         print(f"Route {route_id} model trained successfully ({training_time:.2f}s)")
# #         return model
    
# #     def train_all_models(self, data_path='/app/backend/data/clean_data.csv'):
# #         """Train models for all routes"""
# #         print("Loading clean data for training...")
        
# #         df = pd.read_csv(data_path)
# #         df['ds'] = pd.to_datetime(df['ds'])
        
# #         # Get unique routes
# #         routes = df['route_id'].unique()
# #         print(f"Training models for {len(routes)} routes...")
        
# #         success_count = 0
# #         for route_id in routes:
# #             try:
# #                 model = self.train_route_model(df, route_id)
# #                 if model:
# #                     success_count += 1
# #             except Exception as e:
# #                 print(f"Failed to train model for Route {route_id}: {str(e)}")
        
# #         print(f"Successfully trained {success_count}/{len(routes)} models")
# #         return success_count > 0
    
# #     def generate_forecast(self, route_id, hours_ahead=24, external_regressors=None):
# #         """Generate forecast for specific route"""
# #         if route_id not in self.models:
# #             raise ValueError(f"No trained model found for Route {route_id}")
        
# #         model = self.models[route_id]
        
# #         # Create future dataframe
# #         last_date = datetime.now().replace(minute=0, second=0, microsecond=0)
# #         future_dates = pd.date_range(
# #             start=last_date + timedelta(hours=1),
# #             periods=hours_ahead,
# #             freq='H'
# #         )
        
# #         future_df = pd.DataFrame({
# #             'ds': future_dates
# #         })
        
# #         # Add time-based regressors
# #         future_df['hour'] = future_df['ds'].dt.hour
# #         future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
# #         future_df['is_rush_hour'] = future_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
# #         # Add weather regressors (use defaults if not provided)
# #         if external_regressors is not None:
# #             for col in ['weather_factor', 'temp_extreme', 'heavy_rain']:
# #                 if col in external_regressors:
# #                     future_df[col] = external_regressors[col][:len(future_df)]
# #                 else:
# #                     # Default values
# #                     if col == 'weather_factor':
# #                         future_df[col] = 1.0
# #                     else:
# #                         future_df[col] = 0
# #         else:
# #             future_df['weather_factor'] = 1.0
# #             future_df['temp_extreme'] = 0
# #             future_df['heavy_rain'] = 0
        
# #         # Generate forecast
# #         forecast = model.predict(future_df)
        
# #         # Ensure non-negative predictions
# #         forecast['yhat'] = forecast['yhat'].clip(lower=0)
# #         forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
# #         forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
# #         return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
# #     def evaluate_model_performance(self, route_id, data_path='/app/backend/data/clean_data.csv'):
# #         """Evaluate model performance using cross-validation"""
# #         if route_id not in self.models:
# #             print(f"No model found for Route {route_id}")
# #             return None
        
# #         print(f"Evaluating model performance for Route {route_id}...")
        
# #         # Load data
# #         df = pd.read_csv(data_path)
# #         df['ds'] = pd.to_datetime(df['ds'])
        
# #         # Prepare route data
# #         route_df = self.prepare_data_for_training(df, route_id)
        
# #         if len(route_df) < 72:  # Need at least 3 days for CV
# #             print(f"Insufficient data for cross-validation: {len(route_df)} records")
# #             return None
        
# #         try:
# #             model = self.models[route_id]
            
# #             # Perform cross-validation
# #             cv_results = cross_validation(
# #                 model,
# #                 initial='48 hours',
# #                 period='12 hours',
# #                 horizon='24 hours'
# #             )
            
# #             # Calculate performance metrics
# #             metrics = performance_metrics(cv_results)
            
# #             performance = {
# #                 'mae': metrics['mae'].mean(),
# #                 'rmse': metrics['rmse'].mean(),
# #                 'mape': metrics['mape'].mean(),
# #                 'coverage': ((cv_results['y'] >= cv_results['yhat_lower']) & 
# #                            (cv_results['y'] <= cv_results['yhat_upper'])).mean()
# #             }
            
# #             print(f"Route {route_id} Performance - MAE: {performance['mae']:.2f}, RMSE: {performance['rmse']:.2f}")
# #             return performance
            
# #         except Exception as e:
# #             print(f"Cross-validation failed for Route {route_id}: {str(e)}")
# #             return None
    
# #     def save_models(self, model_dir='/app/backend/models'):
# #         """Save trained models to disk"""
# #         model_path = Path(model_dir)
# #         model_path.mkdir(exist_ok=True)
        
# #         for route_id, model in self.models.items():
# #             model_file = model_path / f'prophet_route_{route_id}.pkl'
            
# #             # Save model with metadata
# #             model_data = {
# #                 'model': model,
# #                 'metadata': self.model_metadata.get(route_id, {}),
# #                 'regressors': self.regressors
# #             }
            
# #             joblib.dump(model_data, model_file)
# #             print(f"Saved model for Route {route_id}")
        
# #         # Save overall metadata
# #         metadata_file = model_path / 'models_metadata.json'
# #         with open(metadata_file, 'w') as f:
# #             json.dump(self.model_metadata, f, indent=2)
        
# #         print(f"All models saved to {model_dir}")
    
# #     def load_models(self, model_dir='/app/backend/models'):
# #         """Load trained models from disk"""
# #         model_path = Path(model_dir)
        
# #         if not model_path.exists():
# #             print(f"Model directory {model_dir} does not exist")
# #             return False
        
# #         model_files = list(model_path.glob('prophet_route_*.pkl'))
        
# #         if not model_files:
# #             print("No model files found")
# #             return False
        
# #         loaded_count = 0
# #         for model_file in model_files:
# #             try:
# #                 # Extract route ID from filename
# #                 route_id = int(model_file.stem.split('_')[-1])
                
# #                 # Load model
# #                 model_data = joblib.load(model_file)
# #                 self.models[route_id] = model_data['model']
# #                 self.model_metadata[route_id] = model_data.get('metadata', {})
                
# #                 loaded_count += 1
# #                 print(f"Loaded model for Route {route_id}")
                
# #             except Exception as e:
# #                 print(f"Failed to load {model_file}: {str(e)}")
        
# #         print(f"Loaded {loaded_count} models successfully")
# #         return loaded_count > 0

# # def main():
# #     """Main function to train and save models"""
# #     print("Starting Prophet model training...")
    
# #     forecaster = BusRidershipForecaster()
    
# #     # Train models
# #     success = forecaster.train_all_models()
    
# #     if success:
# #         # Save models
# #         forecaster.save_models()
        
# #         # Evaluate performance for one route as example
# #         performance = forecaster.evaluate_model_performance(500)
# #         if performance:
# #             print(f"Sample performance metrics: {performance}")
        
# #         print("Model training completed successfully!")
# #     else:
# #         print("Model training failed!")

# # if __name__ == "__main__":
# #     main()

# # from pathlib import Path
# # import pandas as pd
# # import numpy as np
# # from prophet import Prophet
# # from prophet.diagnostics import cross_validation, performance_metrics
# # import joblib
# # import json
# # from datetime import datetime, timedelta
# # import warnings
# # import logging

# # # Suppress Prophet warnings
# # warnings.filterwarnings('ignore')
# # logging.getLogger('prophet').setLevel(logging.WARNING)

# # # Define root directory and data path
# # ROOT_DIR = Path(__file__).parent
# # DATA_PATH = ROOT_DIR / 'data' / 'clean_data.csv'

# # class BusRidershipForecaster:
# #     """Prophet-based forecasting model for bus ridership demand"""
    
# #     def __init__(self):
# #         self.models = {}
# #         self.model_metadata = {}
# #         self.regressors = [
# #             'is_weekend', 'is_rush_hour', 'weather_factor', 
# #             'temp_extreme', 'heavy_rain', 'hour'
# #         ]
        
# #     def prepare_data_for_training(self, df, route_id=None):
# #         """Prepare data for Prophet training"""
# #         if route_id:
# #             df = df[df['route_id'] == route_id].copy()
        
# #         # Ensure required columns exist
# #         required_cols = ['ds', 'y']
# #         for col in required_cols:
# #             if col not in df.columns:
# #                 raise ValueError(f"Missing required column: {col}")
        
# #         # Convert to datetime
# #         df['ds'] = pd.to_datetime(df['ds'])
        
# #         # Sort by timestamp
# #         df = df.sort_values('ds').reset_index(drop=True)
        
# #         # Remove duplicates
# #         df = df.drop_duplicates(subset=['ds'])
        
# #         # Ensure non-negative values
# #         df['y'] = df['y'].clip(lower=0)
        
# #         return df
    
# #     def create_prophet_model(self, route_id):
# #         """Create and configure Prophet model for specific route"""
# #         # [Unchanged, keeping as provided]
# #         route_params = {
# #             500: {  # High-frequency city route
# #                 'changepoint_prior_scale': 0.1,
# #                 'seasonality_prior_scale': 15.0,
# #                 'daily_seasonality': True
# #             },
# #             501: {  # IT corridor route
# #                 'changepoint_prior_scale': 0.05,
# #                 'seasonality_prior_scale': 20.0,
# #                 'daily_seasonality': True
# #             },
# #             502: {  # Tech hub route
# #                 'changepoint_prior_scale': 0.08,
# #                 'seasonality_prior_scale': 25.0,
# #                 'daily_seasonality': True
# #             },
# #             503: {  # Residential route
# #                 'changepoint_prior_scale': 0.05,
# #                 'seasonality_prior_scale': 10.0,
# #                 'daily_seasonality': False
# #             },
# #             504: {  # Commercial route
# #                 'changepoint_prior_scale': 0.07,
# #                 'seasonality_prior_scale': 18.0,
# #                 'daily_seasonality': True
# #             }
# #         }
        
# #         params = route_params.get(route_id, {
# #             'changepoint_prior_scale': 0.05,
# #             'seasonality_prior_scale': 15.0,
# #             'daily_seasonality': True
# #         })
        
# #         model = Prophet(
# #             changepoint_prior_scale=params['changepoint_prior_scale'],
# #             seasonality_prior_scale=params['seasonality_prior_scale'],
# #             holidays_prior_scale=10.0,
# #             daily_seasonality=params['daily_seasonality'],
# #             weekly_seasonality=True,
# #             yearly_seasonality=False,  # Not enough data for yearly
# #             interval_width=0.8
# #         )
        
# #         model.add_seasonality(
# #             name='rush_hour_pattern',
# #             period=24,
# #             fourier_order=8,
# #             condition_name='is_rush_hour'
# #         )
        
# #         regressor_configs = {
# #             'is_weekend': {'mode': 'additive', 'prior_scale': 2.0},
# #             'is_rush_hour': {'mode': 'multiplicative', 'prior_scale': 1.5},
# #             'weather_factor': {'mode': 'multiplicative', 'prior_scale': 1.0},
# #             'temp_extreme': {'mode': 'additive', 'prior_scale': 1.0},
# #             'heavy_rain': {'mode': 'additive', 'prior_scale': 1.5},
# #             'hour': {'mode': 'additive', 'prior_scale': 0.5}
# #         }
        
# #         for regressor, config in regressor_configs.items():
# #             model.add_regressor(
# #                 regressor,
# #                 mode=config['mode'],
# #                 prior_scale=config['prior_scale']
# #             )
        
# #         return model
    
# #     def train_route_model(self, df, route_id):
# #         """Train Prophet model for specific route"""
# #         print(f"Training model for Route {route_id}...")
        
# #         train_df = self.prepare_data_for_training(df, route_id)
        
# #         if len(train_df) < 24:
# #             print(f"Insufficient data for Route {route_id}: {len(train_df)} records")
# #             return None
        
# #         model = self.create_prophet_model(route_id)
        
# #         train_df['is_rush_hour'] = train_df.get('is_rush_hour', 0)
        
# #         start_time = datetime.now()
# #         model.fit(train_df)
# #         training_time = (datetime.now() - start_time).total_seconds()
        
# #         self.models[route_id] = model
# #         self.model_metadata[route_id] = {
# #             'training_date': datetime.now().isoformat(),
# #             'training_records': len(train_df),
# #             'training_time_seconds': training_time,
# #             'data_start': train_df['ds'].min().isoformat(),
# #             'data_end': train_df['ds'].max().isoformat()
# #         }
        
# #         print(f"Route {route_id} model trained successfully ({training_time:.2f}s)")
# #         return model
    
# #     def train_all_models(self, data_path=DATA_PATH):
# #         """Train models for all routes"""
# #         print(f"Loading clean data from {data_path}...")
        
# #         try:
# #             df = pd.read_csv(data_path)
# #             df['ds'] = pd.to_datetime(df['ds'])
# #         except FileNotFoundError:
# #             logging.error(f"Data file not found at {data_path}")
# #             return False
# #         except Exception as e:
# #             logging.error(f"Error loading data: {str(e)}")
# #             return False
        
# #         routes = df['route_id'].unique()
# #         print(f"Training models for {len(routes)} routes...")
        
# #         success_count = 0
# #         for route_id in routes:
# #             try:
# #                 model = self.train_route_model(df, route_id)
# #                 if model:
# #                     success_count += 1
# #             except Exception as e:
# #                 print(f"Failed to train model for Route {route_id}: {str(e)}")
        
# #         print(f"Successfully trained {success_count}/{len(routes)} models")
# #         return success_count > 0
    
# #     def generate_forecast(self, route_id, hours_ahead=24, external_regressors=None):
# #         """Generate forecast for specific route"""
# #         if route_id not in self.models:
# #             raise ValueError(f"No trained model found for Route {route_id}")
        
# #         model = self.models[route_id]
        
# #         last_date = datetime.now().replace(minute=0, second=0, microsecond=0)
# #         future_dates = pd.date_range(
# #             start=last_date + timedelta(hours=1),
# #             periods=hours_ahead,
# #             freq='H'
# #         )
        
# #         future_df = pd.DataFrame({
# #             'ds': future_dates
# #         })
        
# #         future_df['hour'] = future_df['ds'].dt.hour
# #         future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
# #         future_df['is_rush_hour'] = future_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
# #         if external_regressors is not None:
# #             for col in ['weather_factor', 'temp_extreme', 'heavy_rain']:
# #                 if col in external_regressors:
# #                     future_df[col] = external_regressors[col][:len(future_df)]
# #                 else:
# #                     if col == 'weather_factor':
# #                         future_df[col] = 1.0
# #                     else:
# #                         future_df[col] = 0
# #         else:
# #             future_df['weather_factor'] = 1.0
# #             future_df['temp_extreme'] = 0
# #             future_df['heavy_rain'] = 0
        
# #         forecast = model.predict(future_df)
        
# #         forecast['yhat'] = forecast['yhat'].clip(lower=0)
# #         forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
# #         forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
# #         return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
# #     def evaluate_model_performance(self, route_id, data_path=DATA_PATH):
# #         """Evaluate model performance using cross-validation"""
# #         if route_id not in self.models:
# #             print(f"No model found for Route {route_id}")
# #             return None
        
# #         print(f"Evaluating model performance for Route {route_id}...")
        
# #         try:
# #             df = pd.read_csv(data_path)
# #             df['ds'] = pd.to_datetime(df['ds'])
# #         except FileNotFoundError:
# #             logging.error(f"Data file not found at {data_path}")
# #             return None
# #         except Exception as e:
# #             logging.error(f"Error loading data: {str(e)}")
# #             return None
        
# #         route_df = self.prepare_data_for_training(df, route_id)
        
# #         if len(route_df) < 72:
# #             print(f"Insufficient data for cross-validation: {len(route_df)} records")
# #             return None
        
# #         try:
# #             model = self.models[route_id]
            
# #             cv_results = cross_validation(
# #                 model,
# #                 initial='48 hours',
# #                 period='12 hours',
# #                 horizon='24 hours'
# #             )
            
# #             metrics = performance_metrics(cv_results)
            
# #             performance = {
# #                 'mae': metrics['mae'].mean(),
# #                 'rmse': metrics['rmse'].mean(),
# #                 'mape': metrics['mape'].mean(),
# #                 'coverage': ((cv_results['y'] >= cv_results['yhat_lower']) & 
# #                            (cv_results['y'] <= cv_results['yhat_upper'])).mean()
# #             }
            
# #             print(f"Route {route_id} Performance - MAE: {performance['mae']:.2f}, RMSE: {performance['rmse']:.2f}")
# #             return performance
            
# #         except Exception as e:
# #             print(f"Cross-validation failed for Route {route_id}: {str(e)}")
# #             return None
    
# #     def save_models(self, model_dir=ROOT_DIR / 'models'):
# #         """Save trained models to disk"""
# #         model_path = Path(model_dir)
# #         model_path.mkdir(exist_ok=True)
        
# #         for route_id, model in self.models.items():
# #             model_file = model_path / f'prophet_route_{route_id}.pkl'
            
# #             model_data = {
# #                 'model': model,
# #                 'metadata': self.model_metadata.get(route_id, {}),
# #                 'regressors': self.regressors
# #             }
            
# #             joblib.dump(model_data, model_file)
# #             print(f"Saved model for Route {route_id}")
        
# #         metadata_file = model_path / 'models_metadata.json'
# #         with open(metadata_file, 'w') as f:
# #             json.dump(self.model_metadata, f, indent=2)
        
# #         print(f"All models saved to {model_dir}")
    
# #     def load_models(self, model_dir=ROOT_DIR / 'models'):
# #         """Load trained models from disk"""
# #         model_path = Path(model_dir)
        
# #         if not model_path.exists():
# #             print(f"Model directory {model_dir} does not exist")
# #             return False
        
# #         model_files = list(model_path.glob('prophet_route_*.pkl'))
        
# #         if not model_files:
# #             print("No model files found")
# #             return False
        
# #         loaded_count = 0
# #         for model_file in model_files:
# #             try:
# #                 route_id = int(model_file.stem.split('_')[-1])
                
# #                 model_data = joblib.load(model_file)
# #                 self.models[route_id] = model_data['model']
# #                 self.model_metadata[route_id] = model_data.get('metadata', {})
                
# #                 loaded_count += 1
# #                 print(f"Loaded model for Route {route_id}")
                
# #             except Exception as e:
# #                 print(f"Failed to load {model_file}: {str(e)}")
        
# #         print(f"Loaded {loaded_count} models successfully")
# #         return loaded_count > 0

# # def main():
# #     """Main function to train and save models"""
# #     print("Starting Prophet model training...")
    
# #     forecaster = BusRidershipForecaster()
    
# #     success = forecaster.train_all_models()
    
# #     if success:
# #         forecaster.save_models()
        
# #         performance = forecaster.evaluate_model_performance(500)
# #         if performance:
# #             print(f"Sample performance metrics: {performance}")
        
# #         print("Model training completed successfully!")
# #     else:
# #         print("Model training failed!")

# # if __name__ == "__main__":
# #     main()
# #!/usr/bin/env python3
# """
# Smart Bus Management System - Prophet Forecasting Model
# Implements demand prediction using Facebook Prophet with external regressors
# """

# # import pandas as pd
# # import numpy as np
# # from prophet import Prophet
# # from prophet.diagnostics import cross_validation, performance_metrics
# # import joblib
# # import json
# # from datetime import datetime, timedelta
# # from pathlib import Path
# # import warnings
# # import logging

# # # Suppress Prophet warnings
# # warnings.filterwarnings('ignore')
# # logging.getLogger('prophet').setLevel(logging.WARNING)

# # # Set up logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # class BusRidershipForecaster:
# #     """Prophet-based forecasting model for bus ridership demand"""
    
# #     def __init__(self):
# #         self.models = {}
# #         self.model_metadata = {}
# #         self.regressors = [
# #             'is_weekend', 'is_rush_hour', 'weather_factor', 
# #             'temp_extreme', 'heavy_rain', 'hour'
# #         ]
        
# #     def prepare_data_for_training(self, df, route_id=None):
# #         """Prepare data for Prophet training"""
# #         logger.info(f"Preparing data for route_id={route_id}")
# #         if route_id:
# #             df = df[df['route_id'] == route_id].copy()
        
# #         # Ensure required columns exist
# #         required_cols = ['ds', 'y']
# #         for col in required_cols:
# #             if col not in df.columns:
# #                 raise ValueError(f"Missing required column: {col}")
        
# #         # Add defaults for missing regressors and ensure correct types
# #         for col in self.regressors:
# #             if col not in df.columns:
# #                 if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
# #                     df[col] = False  # Default to boolean False
# #                 elif col == 'weather_factor':
# #                     df[col] = 1.0
# #                 elif col == 'hour':
# #                     df[col] = df['ds'].dt.hour.astype(float)
# #             else:
# #                 if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
# #                     df[col] = df[col].astype(bool)  # Convert to boolean
# #                 else:
# #                     df[col] = df[col].astype(float)  # Ensure numeric
        
# #         # Select only necessary columns to exclude extras like strings
# #         columns_to_keep = required_cols + self.regressors
# #         df = df[columns_to_keep]
        
# #         # Convert to datetime
# #         df['ds'] = pd.to_datetime(df['ds'])
        
# #         # Sort by timestamp
# #         df = df.sort_values('ds').reset_index(drop=True)
        
# #         # Remove duplicates
# #         df = df.drop_duplicates(subset=['ds'])
        
# #         # Ensure non-negative values
# #         df['y'] = df['y'].clip(lower=0).astype(float)
        
# #         # Log data info for debugging
# #         logger.info(f"Prepared data: {len(df)} rows, columns: {df.columns.tolist()}")
# #         logger.info(f"Data types: {df.dtypes.to_dict()}")
        
# #         return df
    
# #     def create_prophet_model(self, route_id):
# #         """Create and configure Prophet model for specific route"""
        
# #         # Route-specific parameters based on characteristics
# #         route_params = {
# #             500: {  # High-frequency city route
# #                 'changepoint_prior_scale': 0.1,
# #                 'seasonality_prior_scale': 15.0,
# #                 'daily_seasonality': True
# #             },
# #             501: {  # IT corridor route
# #                 'changepoint_prior_scale': 0.05,
# #                 'seasonality_prior_scale': 20.0,
# #                 'daily_seasonality': True
# #             },
# #             502: {  # Tech hub route
# #                 'changepoint_prior_scale': 0.08,
# #                 'seasonality_prior_scale': 25.0,
# #                 'daily_seasonality': True
# #             },
# #             503: {  # Residential route
# #                 'changepoint_prior_scale': 0.05,
# #                 'seasonality_prior_scale': 10.0,
# #                 'daily_seasonality': False
# #             },
# #             504: {  # Commercial route
# #                 'changepoint_prior_scale': 0.07,
# #                 'seasonality_prior_scale': 18.0,
# #                 'daily_seasonality': True
# #             }
# #         }
        
# #         params = route_params.get(route_id, {
# #             'changepoint_prior_scale': 0.05,
# #             'seasonality_prior_scale': 15.0,
# #             'daily_seasonality': True
# #         })
        
# #         model = Prophet(
# #             changepoint_prior_scale=params['changepoint_prior_scale'],
# #             seasonality_prior_scale=params['seasonality_prior_scale'],
# #             holidays_prior_scale=10.0,
# #             daily_seasonality=params['daily_seasonality'],
# #             weekly_seasonality=True,
# #             yearly_seasonality=False,  # Not enough data for yearly
# #             interval_width=0.8
# #         )
        
# #         # Add custom seasonalities for bus patterns
# #         model.add_seasonality(
# #             name='rush_hour_pattern',
# #             period=24,
# #             fourier_order=8,
# #             condition_name='is_rush_hour'
# #         )
        
# #         # Add external regressors
# #         regressor_configs = {
# #             'is_weekend': {'mode': 'additive', 'prior_scale': 2.0},
# #             'is_rush_hour': {'mode': 'multiplicative', 'prior_scale': 1.5},
# #             'weather_factor': {'mode': 'multiplicative', 'prior_scale': 1.0},
# #             'temp_extreme': {'mode': 'additive', 'prior_scale': 1.0},
# #             'heavy_rain': {'mode': 'additive', 'prior_scale': 1.5},
# #             'hour': {'mode': 'additive', 'prior_scale': 0.5}
# #         }
        
# #         for regressor, config in regressor_configs.items():
# #             model.add_regressor(
# #                 regressor,
# #                 mode=config['mode'],
# #                 prior_scale=config['prior_scale']
# #             )
        
# #         return model
    
# #     def train_route_model(self, df, route_id):
# #         """Train Prophet model for specific route"""
# #         print(f"Training model for Route {route_id}...")
        
# #         # Prepare data
# #         train_df = self.prepare_data_for_training(df, route_id)
        
# #         if len(train_df) < 24:  # Need at least 24 hours of data
# #             print(f"Insufficient data for Route {route_id}: {len(train_df)} records")
# #             return None
        
# #         # Create and configure model
# #         model = self.create_prophet_model(route_id)
        
# #         # Ensure conditional column is boolean
# #         train_df['is_rush_hour'] = train_df['is_rush_hour'].astype(bool)
        
# #         # Train model
# #         start_time = datetime.now()
# #         model.fit(train_df)
# #         training_time = (datetime.now() - start_time).total_seconds()
        
# #         # Store model and metadata
# #         self.models[route_id] = model
# #         self.model_metadata[route_id] = {
# #             'training_date': datetime.now().isoformat(),
# #             'training_records': len(train_df),
# #             'training_time_seconds': training_time,
# #             'data_start': train_df['ds'].min().isoformat(),
# #             'data_end': train_df['ds'].max().isoformat()
# #         }
        
# #         print(f"Route {route_id} model trained successfully ({training_time:.2f}s)")
# #         return model
    
# #     def train_all_models(self, data_path='/app/backend/data/clean_data.csv'):
# #         """Train models for all routes"""
# #         print("Loading clean data for training...")
        
# #         df = pd.read_csv(data_path)
# #         df['ds'] = pd.to_datetime(df['ds'])
        
# #         # Get unique routes
# #         routes = df['route_id'].unique()
# #         print(f"Training models for {len(routes)} routes...")
        
# #         success_count = 0
# #         for route_id in routes:
# #             try:
# #                 model = self.train_route_model(df, route_id)
# #                 if model:
# #                     success_count += 1
# #             except Exception as e:
# #                 print(f"Failed to train model for Route {route_id}: {str(e)}")
        
# #         print(f"Successfully trained {success_count}/{len(routes)} models")
# #         return success_count > 0
    
# #     def generate_forecast(self, route_id, hours_ahead=24, external_regressors=None):
# #         """Generate forecast for specific route"""
# #         if route_id not in self.models:
# #             raise ValueError(f"No trained model found for Route {route_id}")
        
# #         model = self.models[route_id]
        
# #         # Create future dataframe
# #         last_date = datetime.now().replace(minute=0, second=0, microsecond=0)
# #         future_dates = pd.date_range(
# #             start=last_date + timedelta(hours=1),
# #             periods=hours_ahead,
# #             freq='H'
# #         )
        
# #         future_df = pd.DataFrame({
# #             'ds': future_dates
# #         })
        
# #         # Add time-based regressors
# #         future_df['hour'] = future_df['ds'].dt.hour
# #         future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
# #         future_df['is_rush_hour'] = future_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
# #         # Add weather regressors (use defaults if not provided)
# #         if external_regressors is not None:
# #             for col in ['weather_factor', 'temp_extreme', 'heavy_rain']:
# #                 if col in external_regressors:
# #                     future_df[col] = external_regressors[col][:len(future_df)]
# #                 else:
# #                     # Default values
# #                     if col == 'weather_factor':
# #                         future_df[col] = 1.0
# #                     else:
# #                         future_df[col] = 0
# #         else:
# #             future_df['weather_factor'] = 1.0
# #             future_df['temp_extreme'] = 0
# #             future_df['heavy_rain'] = 0
        
# #         # Generate forecast
# #         forecast = model.predict(future_df)
        
# #         # Ensure non-negative predictions
# #         forecast['yhat'] = forecast['yhat'].clip(lower=0)
# #         forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
# #         forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
# #         return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
# #     def evaluate_model_performance(self, route_id, data_path='/app/backend/data/clean_data.csv'):
# #         """Evaluate model performance using cross-validation"""
# #         if route_id not in self.models:
# #             print(f"No model found for Route {route_id}")
# #             return None
        
# #         print(f"Evaluating model performance for Route {route_id}...")
        
# #         # Load data
# #         df = pd.read_csv(data_path)
# #         df['ds'] = pd.to_datetime(df['ds'])
        
# #         # Prepare route data
# #         route_df = self.prepare_data_for_training(df, route_id)
        
# #         if len(route_df) < 72:  # Need at least 3 days for CV
# #             print(f"Insufficient data for cross-validation: {len(route_df)} records")
# #             return None
        
# #         try:
# #             model = self.models[route_id]
            
# #             # Perform cross-validation
# #             cv_results = cross_validation(
# #                 model,
# #                 initial='48 hours',
# #                 period='12 hours',
# #                 horizon='24 hours'
# #             )
            
# #             # Calculate performance metrics
# #             metrics = performance_metrics(cv_results)
            
# #             performance = {
# #                 'mae': metrics['mae'].mean(),
# #                 'rmse': metrics['rmse'].mean(),
# #                 'mape': metrics['mape'].mean(),
# #                 'coverage': ((cv_results['y'] >= cv_results['yhat_lower']) & 
# #                            (cv_results['y'] <= cv_results['yhat_upper'])).mean()
# #             }
            
# #             print(f"Route {route_id} Performance - MAE: {performance['mae']:.2f}, RMSE: {performance['rmse']:.2f}")
# #             return performance
            
# #         except Exception as e:
# #             print(f"Cross-validation failed for Route {route_id}: {str(e)}")
# #             return None
    
# #     def save_models(self, model_dir='/app/backend/models'):
# #         """Save trained models to disk"""
# #         model_path = Path(model_dir)
# #         model_path.mkdir(exist_ok=True)
        
# #         for route_id, model in self.models.items():
# #             model_file = model_path / f'prophet_route_{route_id}.pkl'
            
# #             # Save model with metadata
# #             model_data = {
# #                 'model': model,
# #                 'metadata': self.model_metadata.get(route_id, {}),
# #                 'regressors': self.regressors
# #             }
            
# #             joblib.dump(model_data, model_file)
# #             print(f"Saved model for Route {route_id}")
        
# #         # Save overall metadata
# #         metadata_file = model_path / 'models_metadata.json'
# #         with open(metadata_file, 'w') as f:
# #             json.dump(self.model_metadata, f, indent=2)
        
# #         print(f"All models saved to {model_dir}")
    
# #     def load_models(self, model_dir='/app/backend/models'):
# #         """Load trained models from disk"""
# #         model_path = Path(model_dir)
        
# #         if not model_path.exists():
# #             print(f"Model directory {model_dir} does not exist")
# #             return False
        
# #         model_files = list(model_path.glob('prophet_route_*.pkl'))
        
# #         if not model_files:
# #             print("No model files found")
# #             return False
        
# #         loaded_count = 0
# #         for model_file in model_files:
# #             try:
# #                 # Extract route ID from filename
# #                 route_id = int(model_file.stem.split('_')[-1])
                
# #                 # Load model
# #                 model_data = joblib.load(model_file)
# #                 self.models[route_id] = model_data['model']
# #                 self.model_metadata[route_id] = model_data.get('metadata', {})
                
# #                 loaded_count += 1
# #                 print(f"Loaded model for Route {route_id}")
                
# #             except Exception as e:
# #                 print(f"Failed to load {model_file}: {str(e)}")
        
# #         print(f"Loaded {loaded_count} models successfully")
# #         return loaded_count > 0

# # def main():
# #     """Main function to train and save models"""
# #     print("Starting Prophet model training...")
    
# #     forecaster = BusRidershipForecaster()
    
# #     # Train models
# #     success = forecaster.train_all_models()
    
# #     if success:
# #         # Save models
# #         forecaster.save_models()
        
# #         # Evaluate performance for one route as example
# #         performance = forecaster.evaluate_model_performance(500)
# #         if performance:
# #             print(f"Sample performance metrics: {performance}")
        
# #         print("Model training completed successfully!")
# #     else:
# #         print("Model training failed!")

# # if __name__ == "__main__":
# #     main()

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
#                     df[col] = False  # Default to boolean False
#                 elif col == 'weather_factor':
#                     df[col] = 1.0
#                 elif col == 'hour':
#                     df[col] = df['ds'].dt.hour.astype(float)
#             else:
#                 if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
#                     df[col] = df[col].astype(bool)  # Convert to boolean
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
        
#         # Add custom seasonalities for bus patterns
#         model.add_seasonality(
#             name='rush_hour_pattern',
#             period=24,
#             fourier_order=8,
#             condition_name='is_rush_hour'
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
            
#             # Ensure conditional column is boolean
#             train_df['is_rush_hour'] = train_df['is_rush_hour'].astype(bool)
            
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
        
#         # Add custom seasonalities for bus patterns
#         model.add_seasonality(
#             name='rush_hour_pattern',
#             period=24,
#             fourier_order=8,
#             condition_name='is_rush_hour'
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
            
#             # Ensure conditional column is integer (0/1)
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
#**************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************
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
# import prophet

# # Verify Prophet version
# if prophet.__version__ != '1.1.7':
#     raise ImportError(f"Prophet version 1.1.7 is required, but {prophet.__version__} is installed. Please run `pip install prophet==1.1.7`.")

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
        
#         # Relax cross-validation requirement to 48 records (2 days)
#         if len(route_df) < 48:
#             print(f"Insufficient data for cross-validation: {len(route_df)} records")
#             return None
        
#         try:
#             model = self.models[route_id]
            
#             # Perform cross-validation with relaxed parameters
#             cv_results = cross_validation(
#                 model,
#                 initial='24 hours',  # Reduced from 48 hours
#                 period='12 hours',
#                 horizon='12 hours'   # Reduced from 24 hours
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






# #!/usr/bin/env python3
# """
# Smart Bus Management System - Prophet Forecasting Model with FastAPI
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
# import prophet
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn

# # Verify Prophet version
# if prophet.__version__ != '1.1.7':
#     raise ImportError(f"Prophet version 1.1.7 is required, but {prophet.__version__} is installed. Please run `pip install prophet==1.1.7`.")

# # Suppress Prophet warnings
# warnings.filterwarnings('ignore')
# logging.getLogger('prophet').setLevel(logging.WARNING)

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Define root directory and data path
# ROOT_DIR = Path(__file__).parent
# DATA_PATH = ROOT_DIR / 'data' / 'clean_data.csv'

# # FastAPI app
# app = FastAPI(title="Smart Bus Forecasting API")

# # Pydantic model for request validation
# class ForecastRequest(BaseModel):
#     route_id: str
#     hours_ahead: int
#     include_weather: bool = False

# class BusRidershipForecaster:
#     """Prophet-based forecasting model for bus ridership demand"""
    
#     def __init__(self):
#         self.models = {}
#         self.model_metadata = {}
#         self.regressors = [
#             'is_weekend', 'is_rush_hour', 'weather_factor', 
#             'temp_extreme', 'heavy_rain', 'hour'
#         ]
#         # Load models at startup
#         self.load_models()
    
#     def prepare_data_for_training(self, df, route_id=None):
#         """Prepare data for Prophet training"""
#         logger.info(f"Preparing data for route_id={route_id}")
#         if route_id:
#             df = df[df['route_id'] == route_id].copy()
        
#         required_cols = ['ds', 'y']
#         for col in required_cols:
#             if col not in df.columns:
#                 raise ValueError(f"Missing required column: {col}")
        
#         for col in self.regressors:
#             if col not in df.columns:
#                 if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
#                     df[col] = 0
#                 elif col == 'weather_factor':
#                     df[col] = 1.0
#                 elif col == 'hour':
#                     df[col] = df['ds'].dt.hour.astype(float)
#             else:
#                 if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
#                     df[col] = df[col].astype(int)
#                 else:
#                     df[col] = df[col].astype(float)
        
#         columns_to_keep = required_cols + self.regressors
#         df = df[columns_to_keep]
#         df['ds'] = pd.to_datetime(df['ds'])
#         df = df.sort_values('ds').reset_index(drop=True)
#         df = df.drop_duplicates(subset=['ds'])
#         df['y'] = df['y'].clip(lower=0).astype(float)
        
#         logger.info(f"Prepared data: {len(df)} rows, columns: {df.columns.tolist()}")
#         logger.info(f"Data types: {df.dtypes.to_dict()}")
#         logger.info(f"Sample data:\n{df.head().to_dict()}")
        
#         return df
    
#     def create_prophet_model(self, route_id):
#         """Create and configure Prophet model for specific route"""
#         route_params = {
#             500: {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 15.0, 'daily_seasonality': True},
#             501: {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 20.0, 'daily_seasonality': True},
#             502: {'changepoint_prior_scale': 0.08, 'seasonality_prior_scale': 25.0, 'daily_seasonality': True},
#             503: {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'daily_seasonality': False},
#             504: {'changepoint_prior_scale': 0.07, 'seasonality_prior_scale': 18.0, 'daily_seasonality': True}
#         }
        
#         params = route_params.get(int(route_id), {
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
#             yearly_seasonality=False,
#             interval_width=0.8
#         )
        
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
#             route_id = int(route_id)
#             train_df = self.prepare_data_for_training(df, route_id)
            
#             if len(train_df) < 24:
#                 print(f"Insufficient data for Route {route_id}: {len(train_df)} records")
#                 return None
            
#             model = self.create_prophet_model(route_id)
#             train_df['is_rush_hour'] = train_df['is_rush_hour'].astype(int)
            
#             logger.info(f"Training data for Route {route_id}:\n{train_df.head().to_dict()}")
            
#             start_time = datetime.now()
#             model.fit(train_df)
#             training_time = (datetime.now() - start_time).total_seconds()
            
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
#             df['route_id'] = df['route_id'].astype(int)
#         except FileNotFoundError:
#             logging.error(f"Data file not found at {data_path}")
#             return False
#         except Exception as e:
#             logging.error(f"Error loading data: {str(e)}")
#             return False
        
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
#         route_id = int(route_id)
#         if route_id not in self.models:
#             raise ValueError(f"No trained model found for Route {route_id}")
        
#         model = self.models[route_id]
        
#         last_date = datetime.now().replace(minute=0, second=0, microsecond=0)
#         future_dates = pd.date_range(
#             start=last_date + timedelta(hours=1),
#             periods=hours_ahead,
#             freq='H'
#         )
        
#         future_df = pd.DataFrame({'ds': future_dates})
#         future_df['hour'] = future_df['ds'].dt.hour
#         future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
#         future_df['is_rush_hour'] = future_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
#         if external_regressors is not None:
#             for col in ['weather_factor', 'temp_extreme', 'heavy_rain']:
#                 if col in external_regressors:
#                     future_df[col] = external_regressors[col][:len(future_df)]
#                 else:
#                     future_df[col] = 1.0 if col == 'weather_factor' else 0
#         else:
#             future_df['weather_factor'] = 1.0
#             future_df['temp_extreme'] = 0
#             future_df['heavy_rain'] = 0
        
#         forecast = model.predict(future_df)
#         forecast['yhat'] = forecast['yhat'].clip(lower=0)
#         forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
#         forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
#         return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
#     def evaluate_model_performance(self, route_id, data_path=DATA_PATH):
#         """Evaluate model performance using cross-validation"""
#         route_id = int(route_id)
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
        
#         if len(route_df) < 48:
#             print(f"Insufficient data for cross-validation: {len(route_df)} records")
#             return None
        
#         try:
#             model = self.models[route_id]
#             cv_results = cross_validation(
#                 model,
#                 initial='24 hours',
#                 period='12 hours',
#                 horizon='12 hours'
#             )
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
#             route_id = int(route_id)
#             model_file = model_path / f'prophet_route_{route_id}.pkl'
            
#             model_data = {
#                 'model': model,
#                 'metadata': self.model_metadata.get(route_id, {}),
#                 'regressors': self.regressors
#             }
            
#             joblib.dump(model_data, model_file)
#             print(f"Saved model for Route {route_id}")
        
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
#                 route_id = int(model_file.stem.split('_')[-1])
#                 model_data = joblib.load(model_file)
#                 self.models[route_id] = model_data['model']
#                 self.model_metadata[route_id] = model_data.get('metadata', {})
#                 loaded_count += 1
#                 print(f"Loaded model for Route {route_id}")
#             except Exception as e:
#                 print(f"Failed to load {model_file}: {str(e)}")
        
#         print(f"Loaded {loaded_count} models successfully")
#         return loaded_count > 0

# # Initialize forecaster
# forecaster = BusRidershipForecaster()

# @app.post("/api/forecast")
# async def generate_forecast_endpoint(request: ForecastRequest):
#     """API endpoint to generate forecast for a specific route"""
#     try:
#         # Convert route_id to int and validate
#         route_id = int(request.route_id)
#         hours_ahead = request.hours_ahead
        
#         # Check if model exists, train if not
#         if route_id not in forecaster.models:
#             logger.info(f"No model found for Route {route_id}, attempting to train...")
#             success = forecaster.train_all_models()
#             if not success or route_id not in forecaster.models:
#                 raise HTTPException(status_code=404, detail=f"No trained model for Route {route_id}")
        
#         # Generate forecast
#         external_regressors = None
#         if request.include_weather:
#             # Mock weather data (replace with actual weather API if available)
#             external_regressors = {
#                 'weather_factor': [1.0] * hours_ahead,
#                 'temp_extreme': [0] * hours_ahead,
#                 'heavy_rain': [0] * hours_ahead
#             }
        
#         forecast_df = forecaster.generate_forecast(route_id, hours_ahead, external_regressors)
        
#         # Format response to match frontend expectations
#         predictions = [
#             {
#                 "timestamp": row['ds'].isoformat(),
#                 "predicted_ridership": round(row['yhat'], 2),
#                 "lower_bound": round(row['yhat_lower'], 2),
#                 "upper_bound": round(row['yhat_upper'], 2),
#                 "trend": round(row['trend'], 2) if pd.notnull(row['trend']) else 0
#             }
#             for _, row in forecast_df.iterrows()
#         ]
        
#         return {"predictions": predictions}
    
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logger.error(f"Error generating forecast: {str(e)}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# def main():
#     """Main function to train and save models"""
#     print("Starting Prophet model training...")
    
#     success = forecaster.train_all_models()
    
#     if success:
#         forecaster.save_models()
#         performance = forecaster.evaluate_model_performance(500)
#         if performance:
#             print(f"Sample performance metrics: {performance}")
#         print("Model training completed successfully!")
#     else:
#         print("Model training failed!")

# if __name__ == "__main__":
#     # Run FastAPI server
#     uvicorn.run(app, host="0.0.0.0", port=8000)



#!/usr/bin/env python3
"""
Smart Bus Management System - Prophet Forecasting Model with FastAPI
Implements demand prediction using Facebook Prophet with external regressors
"""
#!/usr/bin/env python3
"""
Smart Bus Management System - Prophet Forecasting Model with FastAPI
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
# import prophet
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn
# from fastapi.middleware.cors import CORSMiddleware

# # Verify Prophet version
# if prophet.__version__ != '1.1.7':
#     raise ImportError(f"Prophet version 1.1.7 is required, but {prophet.__version__} is installed. Please run `pip install prophet==1.1.7`.")

# # Suppress Prophet warnings
# warnings.filterwarnings('ignore')
# logging.getLogger('prophet').setLevel(logging.WARNING)

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Define root directory and data path
# ROOT_DIR = Path(__file__).parent
# DATA_PATH = ROOT_DIR / 'data' / 'clean_data.csv'

# # FastAPI app
# app = FastAPI(title="Smart Bus Forecasting API")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Updated to match frontend port
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods
#     allow_headers=["*"],  # Allow all headers
# )

# # Pydantic models
# class ForecastRequest(BaseModel):
#     route_id: str
#     hours_ahead: int
#     include_weather: bool = False

# class Route(BaseModel):
#     route_id: str
#     source: str
#     destination: str

# class BusRidershipForecaster:
#     """Prophet-based forecasting model for bus ridership demand"""
    
#     def __init__(self):
#         self.models = {}
#         self.model_metadata = {}
#         self.regressors = [
#             'is_weekend', 'is_rush_hour', 'weather_factor', 
#             'temp_extreme', 'heavy_rain', 'hour'
#         ]
#         # Load models at startup
#         self.load_models()
    
#     def prepare_data_for_training(self, df, route_id=None):
#         """Prepare data for Prophet training"""
#         logger.info(f"Preparing data for route_id={route_id}")
#         if route_id:
#             df = df[df['route_id'] == route_id].copy()
        
#         required_cols = ['ds', 'y']
#         for col in required_cols:
#             if col not in df.columns:
#                 raise ValueError(f"Missing required column: {col}")
        
#         for col in self.regressors:
#             if col not in df.columns:
#                 if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
#                     df[col] = 0
#                 elif col == 'weather_factor':
#                     df[col] = 1.0
#                 elif col == 'hour':
#                     df[col] = df['ds'].dt.hour.astype(float)
#             else:
#                 if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
#                     df[col] = df[col].astype(int)
#                 else:
#                     df[col] = df[col].astype(float)
        
#         columns_to_keep = required_cols + self.regressors
#         df = df[columns_to_keep]
#         df['ds'] = pd.to_datetime(df['ds'])
#         df = df.sort_values('ds').reset_index(drop=True)
#         df = df.drop_duplicates(subset=['ds'])
#         df['y'] = df['y'].clip(lower=0).astype(float)
        
#         logger.info(f"Prepared data: {len(df)} rows, columns: {df.columns.tolist()}")
#         logger.info(f"Data types: {df.dtypes.to_dict()}")
#         logger.info(f"Sample data:\n{df.head().to_dict()}")
        
#         return df
    
#     def create_prophet_model(self, route_id):
#         """Create and configure Prophet model for specific route"""
#         route_params = {
#             500: {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 15.0, 'daily_seasonality': True},
#             501: {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 20.0, 'daily_seasonality': True},
#             502: {'changepoint_prior_scale': 0.08, 'seasonality_prior_scale': 25.0, 'daily_seasonality': True},
#             503: {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'daily_seasonality': False},
#             504: {'changepoint_prior_scale': 0.07, 'seasonality_prior_scale': 18.0, 'daily_seasonality': True}
#         }
        
#         params = route_params.get(int(route_id), {
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
#             yearly_seasonality=False,
#             interval_width=0.8
#         )
        
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
#             route_id = int(route_id)
#             train_df = self.prepare_data_for_training(df, route_id)
            
#             if len(train_df) < 24:
#                 print(f"Insufficient data for Route {route_id}: {len(train_df)} records")
#                 return None
            
#             model = self.create_prophet_model(route_id)
#             train_df['is_rush_hour'] = train_df['is_rush_hour'].astype(int)
            
#             logger.info(f"Training data for Route {route_id}:\n{train_df.head().to_dict()}")
            
#             start_time = datetime.now()
#             model.fit(train_df)
#             training_time = (datetime.now() - start_time).total_seconds()
            
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
#             df['route_id'] = df['route_id'].astype(int)
#         except FileNotFoundError:
#             logging.error(f"Data file not found at {data_path}")
#             return False
#         except Exception as e:
#             logging.error(f"Error loading data: {str(e)}")
#             return False
        
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
#         route_id = int(route_id)
#         if route_id not in self.models:
#             raise ValueError(f"No trained model found for Route {route_id}")
        
#         model = self.models[route_id]
        
#         last_date = datetime.now().replace(minute=0, second=0, microsecond=0)
#         future_dates = pd.date_range(
#             start=last_date + timedelta(hours=1),
#             periods=hours_ahead,
#             freq='H'
#         )
        
#         future_df = pd.DataFrame({'ds': future_dates})
#         future_df['hour'] = future_df['ds'].dt.hour
#         future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
#         future_df['is_rush_hour'] = future_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
#         if external_regressors is not None:
#             for col in ['weather_factor', 'temp_extreme', 'heavy_rain']:
#                 if col in external_regressors:
#                     future_df[col] = external_regressors[col][:len(future_df)]
#                 else:
#                     future_df[col] = 1.0 if col == 'weather_factor' else 0
#         else:
#             future_df['weather_factor'] = 1.0
#             future_df['temp_extreme'] = 0
#             future_df['heavy_rain'] = 0
        
#         forecast = model.predict(future_df)
#         forecast['yhat'] = forecast['yhat'].clip(lower=0)
#         forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
#         forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
#         return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
#     def evaluate_model_performance(self, route_id, data_path=DATA_PATH):
#         """Evaluate model performance using cross-validation"""
#         route_id = int(route_id)
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
        
#         if len(route_df) < 48:
#             print(f"Insufficient data for cross-validation: {len(route_df)} records")
#             return None
        
#         try:
#             model = self.models[route_id]
#             cv_results = cross_validation(
#                 model,
#                 initial='24 hours',
#                 period='12 hours',
#                 horizon='12 hours'
#             )
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
#             route_id = int(route_id)
#             model_file = model_path / f'prophet_route_{route_id}.pkl'
            
#             model_data = {
#                 'model': model,
#                 'metadata': self.model_metadata.get(route_id, {}),
#                 'regressors': self.regressors
#             }
            
#             joblib.dump(model_data, model_file)
#             print(f"Saved model for Route {route_id}")
        
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
#                 route_id = int(model_file.stem.split('_')[-1])
#                 model_data = joblib.load(model_file)
#                 self.models[route_id] = model_data['model']
#                 self.model_metadata[route_id] = model_data.get('metadata', {})
#                 loaded_count += 1
#                 print(f"Loaded model for Route {route_id}")
#             except Exception as e:
#                 print(f"Failed to load {model_file}: {str(e)}")
        
#         print(f"Loaded {loaded_count} models successfully")
#         return loaded_count > 0
    
#     def get_routes(self):
#         """Return list of available routes with metadata"""
#         # Mock route metadata (replace with actual data source if available)
#         routes = [
#             {"route_id": "500", "source": "City Center", "destination": "Downtown"},
#             {"route_id": "504", "source": "Commercial Hub", "destination": "Suburb"}
#         ]
#         return [Route(**route) for route in routes]

# # Initialize forecaster
# forecaster = BusRidershipForecaster()

# @app.post("/api/forecast")
# async def generate_forecast_endpoint(request: ForecastRequest):
#     """API endpoint to generate forecast for a specific route"""
#     try:
#         route_id = int(request.route_id)
#         hours_ahead = request.hours_ahead
        
#         if route_id not in forecaster.models:
#             logger.info(f"No model found for Route {route_id}, attempting to train...")
#             success = forecaster.train_all_models()
#             if not success or route_id not in forecaster.models:
#                 raise HTTPException(status_code=404, detail=f"No trained model for Route {route_id}")
        
#         external_regressors = None
#         if request.include_weather:
#             external_regressors = {
#                 'weather_factor': [1.0] * hours_ahead,
#                 'temp_extreme': [0] * hours_ahead,
#                 'heavy_rain': [0] * hours_ahead
#             }
        
#         forecast_df = forecaster.generate_forecast(route_id, hours_ahead, external_regressors)
        
#         predictions = [
#             {
#                 "timestamp": row['ds'].isoformat(),
#                 "predicted_ridership": round(row['yhat'], 2),
#                 "lower_bound": round(row['yhat_lower'], 2),
#                 "upper_bound": round(row['yhat_upper'], 2),
#                 "trend": round(row['trend'], 2) if pd.notnull(row['trend']) else 0
#             }
#             for _, row in forecast_df.iterrows()
#         ]
        
#         return {"predictions": predictions}
    
#     except ValueError as e:
#         logger.error(f"ValueError in forecast endpoint: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=400, detail=str(e))
#     except ConnectionError as e:
#         logger.error(f"Connection error in forecast endpoint: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=503, detail="Service temporarily unavailable")
#     except Exception as e:
#         logger.error(f"Unexpected error in forecast endpoint: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal server error")

# @app.get("/api/routes", response_model=list[Route])
# async def get_routes_endpoint():
#     """API endpoint to get available routes"""
#     try:
#         return forecaster.get_routes()
#     except Exception as e:
#         logger.error(f"Error fetching routes: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal server error")

# def main():
#     """Main function to train and save models"""
#     print("Starting Prophet model training...")
    
#     success = forecaster.train_all_models()
    
#     if success:
#         forecaster.save_models()
#         performance = forecaster.evaluate_model_performance(500)
#         if performance:
#             print(f"Sample performance metrics: {performance}")
#         print("Model training completed successfully!")
#     else:
#         print("Model training failed!")

# if __name__ == "__main__":
#     # Run FastAPI server
#     import sys
#     try:
#         uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
#     except Exception as e:
#         logger.error(f"Server crashed: {str(e)}", exc_info=True)
#         sys.exit(1)

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import joblib
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import logging
import prophet
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Verify Prophet version
if prophet.__version__ != '1.1.7':
    raise ImportError(f"Prophet version 1.1.7 is required, but {prophet.__version__} is installed. Please run `pip install prophet==1.1.7`.")

# Suppress Prophet warnings
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define root directory and data path
ROOT_DIR = Path(__file__).parent
DATA_PATH = ROOT_DIR / 'data' / 'clean_data.csv'

# FastAPI app
app = FastAPI(title="Smart Bus Forecasting API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Updated to match frontend port
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Pydantic models
class ForecastRequest(BaseModel):
    route_id: str
    hours_ahead: int
    include_weather: bool = False

class Route(BaseModel):
    route_id: str
    source: str
    destination: str

class BusRidershipForecaster:
    """Prophet-based forecasting model for bus ridership demand"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.regressors = [
            'is_weekend', 'is_rush_hour', 'weather_factor', 
            'temp_extreme', 'heavy_rain', 'hour'
        ]
        # Load models at startup
        print(f"Loading models at startup from {ROOT_DIR / 'models'}")
        self.load_models()
        if not self.models:
            print(f"No models loaded, attempting to train all models...")
            self.train_all_models()
    
    def prepare_data_for_training(self, df, route_id=None):
        """Prepare data for Prophet training"""
        logger.info(f"Preparing data for route_id={route_id}")
        if route_id:
            df = df[df['route_id'] == route_id].copy()
        
        required_cols = ['ds', 'y']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        for col in self.regressors:
            if col not in df.columns:
                if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
                    df[col] = 0
                elif col == 'weather_factor':
                    df[col] = 1.0
                elif col == 'hour':
                    df[col] = df['ds'].dt.hour.astype(float)
            else:
                if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
                    df[col] = df[col].astype(int)
                else:
                    df[col] = df[col].astype(float)
        
        columns_to_keep = required_cols + self.regressors
        df = df[columns_to_keep]
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        df = df.drop_duplicates(subset=['ds'])
        df['y'] = df['y'].clip(lower=0).astype(float)
        
        logger.info(f"Prepared data: {len(df)} rows, columns: {df.columns.tolist()}")
        logger.info(f"Data types: {df.dtypes.to_dict()}")
        logger.info(f"Sample data:\n{df.head().to_dict()}")
        
        return df
    
    def create_prophet_model(self, route_id):
        """Create and configure Prophet model for specific route"""
        # route_params = {
        #     1: {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 15.0, 'daily_seasonality': True},
        #     501: {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 20.0, 'daily_seasonality': True},
        #     502: {'changepoint_prior_scale': 0.08, 'seasonality_prior_scale': 25.0, 'daily_seasonality': True},
        #     503: {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'daily_seasonality': False},
        #     504: {'changepoint_prior_scale': 0.07, 'seasonality_prior_scale': 18.0, 'daily_seasonality': True}
        # }
        # Extract route numbers from busRoutes
        bus_routes = [
            {"Route No.": "1"}, {"Route No.": "4"}, {"Route No.": "5"}, {"Route No.": "14"}, 
            {"Route No.": "15"}, {"Route No.": "16"}, {"Route No.": "17"}, {"Route No.": "18"},
            {"Route No.": "22"}, {"Route No.": "23"}, {"Route No.": "28"}, {"Route No.": "31"},
            {"Route No.": "32"}, {"Route No.": "33"}, {"Route No.": "34"}, {"Route No.": "35"},
            {"Route No.": "36"}, {"Route No.": "37"}, {"Route No.": "38"}, {"Route No.": "40"},
            {"Route No.": "42"}, {"Route No.": "43"}, {"Route No.": "45"}, {"Route No.": "46"},
            {"Route No.": "47"}, {"Route No.": "48"}, {"Route No.": "49"}, {"Route No.": "50"},
            {"Route No.": "52"}, {"Route No.": "54"}, {"Route No.": "56"}, {"Route No.": "58"},
            {"Route No.": "60"}, {"Route No.": "61"}, {"Route No.": "63"}, {"Route No.": "64"},
            {"Route No.": "65"}, {"Route No.": "66"}, {"Route No.": "67"}, {"Route No.": "68"},
            {"Route No.": "69"}, {"Route No.": "70"}, {"Route No.": "72"}, {"Route No.": "74"},
            {"Route No.": "75"}, {"Route No.": "76"}, {"Route No.": "77"}, {"Route No.": "79"},
            {"Route No.": "82"}, {"Route No.": "83"}, {"Route No.": "84"}, {"Route No.": "85"},
            {"Route No.": "87"}, {"Route No.": "88"}, {"Route No.": "90"}, {"Route No.": "96"},
            {"Route No.": "101"}, {"Route No.": "102"}, {"Route No.": "105"}, {"Route No.": "112"},
            {"Route No.": "116"}, {"Route No.": "117"}, {"Route No.": "122"}, {"Route No.": "123 SH"},
            {"Route No.": "125"}, {"Route No.": "126"}, {"Route No.": "127"}, {"Route No.": "128"},
            {"Route No.": "129"}, {"Route No.": "130"}, {"Route No.": "134"}, {"Route No.": "135"},
            {"Route No.": "136"}, {"Route No.": "138"}, {"Route No.": "141"}, {"Route No.": "142"},
            {"Route No.": "143"}, {"Route No.": "144"}, {"Route No.": "145"}, {"Route No.": "147"},
            {"Route No.": "148"}, {"Route No.": "150"}, {"Route No.": "152"}, {"Route No.": "153"},
            {"Route No.": "160"}, {"Route No.": "200"}, {"Route No.": "201"}, {"Route No.": "202"},
            {"Route No.": "203"}, {"Route No.": "300"}, {"Route No.": "301"}, {"Route No.": "400"},
            {"Route No.": "401"}, {"Route No.": "500"}, {"Route No.": "501"}, {"Route No.": "800"},
            {"Route No.": "900"}
        ]
        route_params = {}
        for route in bus_routes:
            route_no = route["Route No."].replace(" SH", "")  # Remove " SH" if present
            route_params[int(route_no)] = {
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 15.0,
                "daily_seasonality": True
            }

        # Example output
        print(route_params)

        
        params = route_params.get(int(route_id), {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 15.0,
            'daily_seasonality': True
        })
        
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=10.0,
            daily_seasonality=params['daily_seasonality'],
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=0.8
        )
        
        regressor_configs = {
            'is_weekend': {'mode': 'additive', 'prior_scale': 2.0},
            'is_rush_hour': {'mode': 'multiplicative', 'prior_scale': 1.5},
            'weather_factor': {'mode': 'multiplicative', 'prior_scale': 1.0},
            'temp_extreme': {'mode': 'additive', 'prior_scale': 1.0},
            'heavy_rain': {'mode': 'additive', 'prior_scale': 1.5},
            'hour': {'mode': 'additive', 'prior_scale': 0.5}
        }
        
        for regressor, config in regressor_configs.items():
            model.add_regressor(
                regressor,
                mode=config['mode'],
                prior_scale=config['prior_scale']
            )
        
        return model
    
    def train_route_model(self, df, route_id):
        """Train Prophet model for specific route"""
        print(f"Training model for Route {route_id}...")
        
        try:
            route_id = int(route_id)
            train_df = self.prepare_data_for_training(df, route_id)
            
            if len(train_df) < 24:
                print(f"Insufficient data for Route {route_id}: {len(train_df)} records")
                return None
            
            model = self.create_prophet_model(route_id)
            train_df['is_rush_hour'] = train_df['is_rush_hour'].astype(int)
            
            logger.info(f"Training data for Route {route_id}:\n{train_df.head().to_dict()}")
            
            start_time = datetime.now()
            model.fit(train_df)
            training_time = (datetime.now() - start_time).total_seconds()
            
            self.models[route_id] = model
            self.model_metadata[route_id] = {
                'training_date': datetime.now().isoformat(),
                'training_records': len(train_df),
                'training_time_seconds': training_time,
                'data_start': train_df['ds'].min().isoformat(),
                'data_end': train_df['ds'].max().isoformat()
            }
            
            print(f"Route {route_id} model trained successfully ({training_time:.2f}s)")
            return model
        
        except Exception as e:
            print(f"Failed to train model for Route {route_id}: {str(e)}")
            return None
    
    def train_all_models(self, data_path=DATA_PATH):
        """Train models for all routes"""
        print(f"Loading clean data from {data_path}...")
        
        try:
            logger.info(f"Attempting to read file: {data_path}")
            df = pd.read_csv(data_path)
            df['ds'] = pd.to_datetime(df['ds'])
            df['route_id'] = df['route_id'].astype(int)
        except FileNotFoundError:
            logging.error(f"Data file not found at {data_path}")
            return False
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return False
        
        routes = [int(route_id) for route_id in df['route_id'].unique()]
        print(f"Training models for {len(routes)} routes...")
        
        success_count = 0
        for route_id in routes:
            try:
                model = self.train_route_model(df, route_id)
                if model:
                    success_count += 1
            except Exception as e:
                print(f"Failed to train model for Route {route_id}: {str(e)}")
        
        print(f"Successfully trained {success_count}/{len(routes)} models")
        return success_count > 0
    
    def generate_forecast(self, route_id, hours_ahead=24, external_regressors=None):
        """Generate forecast for specific route"""
        route_id = int(route_id)
        if route_id not in self.models:
            raise ValueError(f"No trained model found for Route {route_id}")
        
        model = self.models[route_id]
        
        last_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        future_dates = pd.date_range(
            start=last_date + timedelta(hours=1),
            periods=hours_ahead,
            freq='H'
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        future_df['hour'] = future_df['ds'].dt.hour
        future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
        future_df['is_rush_hour'] = future_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        if external_regressors is not None:
            for col in ['weather_factor', 'temp_extreme', 'heavy_rain']:
                if col in external_regressors:
                    future_df[col] = external_regressors[col][:len(future_df)]
                else:
                    future_df[col] = 1.0 if col == 'weather_factor' else 0
        else:
            future_df['weather_factor'] = 1.0
            future_df['temp_extreme'] = 0
            future_df['heavy_rain'] = 0
        
        forecast = model.predict(future_df)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
    def evaluate_model_performance(self, route_id, data_path=DATA_PATH):
        """Evaluate model performance using cross-validation"""
        route_id = int(route_id)
        if route_id not in self.models:
            print(f"No model found for Route {route_id}")
            return None
        
        print(f"Evaluating model performance for Route {route_id}...")
        
        try:
            logger.info(f"Attempting to read file for evaluation: {data_path}")
            df = pd.read_csv(data_path)
            df['ds'] = pd.to_datetime(df['ds'])
            df['route_id'] = df['route_id'].astype(int)
        except FileNotFoundError:
            logging.error(f"Data file not found at {data_path}")
            return None
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None
        
        route_df = self.prepare_data_for_training(df, route_id)
        
        if len(route_df) < 48:
            print(f"Insufficient data for cross-validation: {len(route_df)} records")
            return None
        
        try:
            model = self.models[route_id]
            cv_results = cross_validation(
                model,
                initial='24 hours',
                period='12 hours',
                horizon='12 hours'
            )
            metrics = performance_metrics(cv_results)
            
            performance = {
                'mae': metrics['mae'].mean(),
                'rmse': metrics['rmse'].mean(),
                'mape': metrics['mape'].mean(),
                'coverage': ((cv_results['y'] >= cv_results['yhat_lower']) & 
                             (cv_results['y'] <= cv_results['yhat_upper'])).mean()
            }
            
            print(f"Route {route_id} Performance - MAE: {performance['mae']:.2f}, RMSE: {performance['rmse']:.2f}")
            return performance
            
        except Exception as e:
            print(f"Cross-validation failed for Route {route_id}: {str(e)}")
            return None
    
    def save_models(self, model_dir=ROOT_DIR / 'models'):
        """Save trained models to disk"""
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        for route_id, model in self.models.items():
            route_id = int(route_id)
            model_file = model_path / f'prophet_route_{route_id}.pkl'
            
            model_data = {
                'model': model,
                'metadata': self.model_metadata.get(route_id, {}),
                'regressors': self.regressors
            }
            
            joblib.dump(model_data, model_file)
            print(f"Saved model for Route {route_id}")
        
        metadata_file = model_path / 'models_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        print(f"All models saved to {model_dir}")
    
    def load_models(self, model_dir=ROOT_DIR / 'models'):
        """Load trained models from disk"""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            print(f"Model directory {model_dir} does not exist")
            return False
        
        model_files = list(model_path.glob('prophet_route_*.pkl'))
        
        if not model_files:
            print("No model files found")
            return False
        
        loaded_count = 0
        for model_file in model_files:
            try:
                route_id = int(model_file.stem.split('_')[-1])
                model_data = joblib.load(model_file)
                self.models[route_id] = model_data['model']
                self.model_metadata[route_id] = model_data.get('metadata', {})
                loaded_count += 1
                print(f"Loaded model for Route {route_id}")
            except Exception as e:
                print(f"Failed to load {model_file}: {str(e)}")
        
        print(f"Loaded {loaded_count} models successfully")
        return loaded_count > 0
    
    def get_routes(self):
        """Return list of available routes with metadata"""
        try:
            df = pd.read_csv(DATA_PATH)
            df['route_id'] = df['route_id'].astype(int)
            unique_routes = df['route_id'].unique()
            routes = [
                {"route_id": str(route_id), "source": f"Source_{route_id}", "destination": f"Destination_{route_id}"}
                for route_id in unique_routes
            ]
            return [Route(**route) for route in routes]
        except FileNotFoundError:
            logging.error(f"Data file not found at {DATA_PATH}")
            return []
        except Exception as e:
            logging.error(f"Error fetching routes: {str(e)}")
            return []

# Initialize forecaster
forecaster = BusRidershipForecaster()

@app.post("/api/forecast")
async def generate_forecast_endpoint(request: ForecastRequest):
    """API endpoint to generate forecast for a specific route"""
    try:
        route_id = int(request.route_id)
        hours_ahead = request.hours_ahead
        
        if route_id not in forecaster.models:
            logger.info(f"No model found for Route {route_id}, attempting to train...")
            success = forecaster.train_all_models()
            if not success or route_id not in forecaster.models:
                logger.warning(f"Failed to train or find model for Route {route_id}")
                raise HTTPException(status_code=404, detail=f"No trained model for Route {route_id}. Available routes: {list(forecaster.models.keys())}")
        
        external_regressors = None
        if request.include_weather:
            external_regressors = {
                'weather_factor': [1.0] * hours_ahead,
                'temp_extreme': [0] * hours_ahead,
                'heavy_rain': [0] * hours_ahead
            }
        
        forecast_df = forecaster.generate_forecast(route_id, hours_ahead, external_regressors)
        
        predictions = [
            {
                "timestamp": row['ds'].isoformat(),
                "predicted_ridership": round(row['yhat'], 2),
                "lower_bound": round(row['yhat_lower'], 2),
                "upper_bound": round(row['yhat_upper'], 2),
                "trend": round(row['trend'], 2) if pd.notnull(row['trend']) else 0
            }
            for _, row in forecast_df.iterrows()
        ]
        
        return {"predictions": predictions}
    
    except ValueError as e:
        logger.error(f"ValueError in forecast endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        logger.error(f"HTTPException in forecast endpoint: {str(e.detail)}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in forecast endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/routes", response_model=list[Route])
async def get_routes_endpoint():
    """API endpoint to get available routes"""
    try:
        return forecaster.get_routes()
    except Exception as e:
        logger.error(f"Error fetching routes: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

def main():
    """Main function to train and save models"""
    print("Starting Prophet model training...")
    
    success = forecaster.train_all_models()
    
    if success:
        forecaster.save_models()
        performance = forecaster.evaluate_model_performance(500)
        if performance:
            print(f"Sample performance metrics: {performance}")
        print("Model training completed successfully!")
    else:
        print("Model training failed!")

if __name__ == "__main__":
    # Run FastAPI server
    import sys
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
    except Exception as e:
        logger.error(f"Server crashed: {str(e)}", exc_info=True)
        sys.exit(1)
