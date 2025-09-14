#!/usr/bin/env python3
"""
Smart Bus Management System - Prophet Forecasting Model
Implements demand prediction using Facebook Prophet with external regressors
"""

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

# Suppress Prophet warnings
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define root directory and data path
ROOT_DIR = Path(__file__).parent
DATA_PATH = ROOT_DIR / 'data' / 'clean_data.csv'

class BusRidershipForecaster:
    """Prophet-based forecasting model for bus ridership demand"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.regressors = [
            'is_weekend', 'is_rush_hour', 'weather_factor', 
            'temp_extreme', 'heavy_rain', 'hour'
        ]
        
    def prepare_data_for_training(self, df, route_id=None):
        """Prepare data for Prophet training"""
        logger.info(f"Preparing data for route_id={route_id}")
        if route_id:
            df = df[df['route_id'] == route_id].copy()
        
        # Ensure required columns exist
        required_cols = ['ds', 'y']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add defaults for missing regressors and ensure correct types
        for col in self.regressors:
            if col not in df.columns:
                if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
                    df[col] = 0  # Default to 0 (integer)
                elif col == 'weather_factor':
                    df[col] = 1.0
                elif col == 'hour':
                    df[col] = df['ds'].dt.hour.astype(float)
            else:
                if col in ['is_weekend', 'is_rush_hour', 'temp_extreme', 'heavy_rain']:
                    df[col] = df[col].astype(int)  # Convert to integer (0/1)
                else:
                    df[col] = df[col].astype(float)  # Ensure numeric
        
        # Select only necessary columns to exclude extras like strings
        columns_to_keep = required_cols + self.regressors
        df = df[columns_to_keep]
        
        # Convert to datetime
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Sort by timestamp
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ds'])
        
        # Ensure non-negative values
        df['y'] = df['y'].clip(lower=0).astype(float)
        
        # Log data info for debugging
        logger.info(f"Prepared data: {len(df)} rows, columns: {df.columns.tolist()}")
        logger.info(f"Data types: {df.dtypes.to_dict()}")
        logger.info(f"Sample data:\n{df.head().to_dict()}")
        
        return df
    
    def create_prophet_model(self, route_id):
        """Create and configure Prophet model for specific route"""
        route_params = {
            500: {  # High-frequency city route
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 15.0,
                'daily_seasonality': True
            },
            501: {  # IT corridor route
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 20.0,
                'daily_seasonality': True
            },
            502: {  # Tech hub route
                'changepoint_prior_scale': 0.08,
                'seasonality_prior_scale': 25.0,
                'daily_seasonality': True
            },
            503: {  # Residential route
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'daily_seasonality': False
            },
            504: {  # Commercial route
                'changepoint_prior_scale': 0.07,
                'seasonality_prior_scale': 18.0,
                'daily_seasonality': True
            }
        }
        
        params = route_params.get(route_id, {
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
            yearly_seasonality=False,  # Not enough data for yearly
            interval_width=0.8
        )
        
        # Add external regressors
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
            # Convert route_id to Python int to avoid numpy.int64
            route_id = int(route_id)
            
            # Prepare data
            train_df = self.prepare_data_for_training(df, route_id)
            
            if len(train_df) < 24:  # Need at least 24 hours of data
                print(f"Insufficient data for Route {route_id}: {len(train_df)} records")
                return None
            
            # Create and configure model
            model = self.create_prophet_model(route_id)
            
            # Ensure regressors are integer (0/1)
            train_df['is_rush_hour'] = train_df['is_rush_hour'].astype(int)
            
            # Log sample data for debugging
            logger.info(f"Training data for Route {route_id}:\n{train_df.head().to_dict()}")
            
            # Train model
            start_time = datetime.now()
            model.fit(train_df)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store model and metadata
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
            # Convert route_id to Python int
            df['route_id'] = df['route_id'].astype(int)
        except FileNotFoundError:
            logging.error(f"Data file not found at {data_path}")
            return False
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return False
        
        # Convert route IDs to Python int
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
        route_id = int(route_id)  # Ensure route_id is Python int
        if route_id not in self.models:
            raise ValueError(f"No trained model found for Route {route_id}")
        
        model = self.models[route_id]
        
        # Create future dataframe
        last_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        future_dates = pd.date_range(
            start=last_date + timedelta(hours=1),
            periods=hours_ahead,
            freq='H'
        )
        
        future_df = pd.DataFrame({
            'ds': future_dates
        })
        
        # Add time-based regressors
        future_df['hour'] = future_df['ds'].dt.hour
        future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
        future_df['is_rush_hour'] = future_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Add weather regressors (use defaults if not provided)
        if external_regressors is not None:
            for col in ['weather_factor', 'temp_extreme', 'heavy_rain']:
                if col in external_regressors:
                    future_df[col] = external_regressors[col][:len(future_df)]
                else:
                    # Default values
                    if col == 'weather_factor':
                        future_df[col] = 1.0
                    else:
                        future_df[col] = 0
        else:
            future_df['weather_factor'] = 1.0
            future_df['temp_extreme'] = 0
            future_df['heavy_rain'] = 0
        
        # Generate forecast
        forecast = model.predict(future_df)
        
        # Ensure non-negative predictions
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
    def evaluate_model_performance(self, route_id, data_path=DATA_PATH):
        """Evaluate model performance using cross-validation"""
        route_id = int(route_id)  # Ensure route_id is Python int
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
        
        # Relax cross-validation requirement to 48 records (2 days)
        if len(route_df) < 48:
            print(f"Insufficient data for cross-validation: {len(route_df)} records")
            return None
        
        try:
            model = self.models[route_id]
            
            # Perform cross-validation with relaxed parameters
            cv_results = cross_validation(
                model,
                initial='24 hours',  # Reduced from 48 hours
                period='12 hours',
                horizon='12 hours'   # Reduced from 24 hours
            )
            
            # Calculate performance metrics
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
            route_id = int(route_id)  # Ensure route_id is Python int
            model_file = model_path / f'prophet_route_{route_id}.pkl'
            
            # Save model with metadata
            model_data = {
                'model': model,
                'metadata': self.model_metadata.get(route_id, {}),
                'regressors': self.regressors
            }
            
            joblib.dump(model_data, model_file)
            print(f"Saved model for Route {route_id}")
        
        # Save overall metadata
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
                # Extract route ID from filename
                route_id = int(model_file.stem.split('_')[-1])
                
                # Load model
                model_data = joblib.load(model_file)
                self.models[route_id] = model_data['model']
                self.model_metadata[route_id] = model_data.get('metadata', {})
                
                loaded_count += 1
                print(f"Loaded model for Route {route_id}")
                
            except Exception as e:
                print(f"Failed to load {model_file}: {str(e)}")
        
        print(f"Loaded {loaded_count} models successfully")
        return loaded_count > 0

def main():
    """Main function to train and save models"""
    print("Starting Prophet model training...")
    
    forecaster = BusRidershipForecaster()
    
    # Train models
    success = forecaster.train_all_models()
    
    if success:
        # Save models
        forecaster.save_models()
        
        # Evaluate performance for one route as example
        performance = forecaster.evaluate_model_performance(500)
        if performance:
            print(f"Sample performance metrics: {performance}")
        
        print("Model training completed successfully!")
    else:
        print("Model training failed!")

if __name__ == "__main__":
    main()