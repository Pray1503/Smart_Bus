#!/usr/bin/env python3
"""
Smart Bus Management System - Data Generation Script
Generates synthetic bus ridership and GPS data for Bangalore routes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random
import math
import uuid

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class BangaloreBusDataGenerator:
    """Generate synthetic bus data for Bangalore routes based on AMTS patterns"""
    
    def __init__(self):
        # Define Bangalore bus routes based on AMTS structure
        self.bangalore_routes = {
            500: {
                'source': 'Majestic Bus Station',
                'destination': 'K.R. Market',
                'distance_km': 8.5,
                'fare': 15,
                'stops': ['Majestic', 'City Railway Station', 'Chickpet', 'Avenue Road', 'K.R. Market']
            },
            501: {
                'source': 'Silk Board',
                'destination': 'Electronic City',
                'distance_km': 12.3,
                'fare': 20,
                'stops': ['Silk Board', 'BTM Layout', 'Bommanahalli', 'Hongasandra', 'Electronic City']
            },
            502: {
                'source': 'Whitefield',
                'destination': 'Marathahalli',
                'distance_km': 15.7,
                'fare': 25,
                'stops': ['Whitefield', 'ITPL', 'Brookefield', 'Kundalahalli', 'Marathahalli']
            },
            503: {
                'source': 'Koramangala',
                'destination': 'Jayanagar',
                'distance_km': 9.2,
                'fare': 18,
                'stops': ['Koramangala', '5th Block', 'Wilson Garden', '4th Block Jayanagar', 'Jayanagar']
            },
            504: {
                'source': 'Indiranagar',
                'destination': 'MG Road',
                'distance_km': 6.8,
                'fare': 12,
                'stops': ['Indiranagar', '100 Feet Road', 'HAL', 'Trinity', 'MG Road']
            }
        }
        
        # Bangalore coordinates (approximate)
        self.city_center = {'lat': 12.9716, 'lng': 77.5946}
        
        # Time periods for ridership patterns
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 8)  # 1 week of data
        
    def generate_gps_coordinates(self, route_id, stop_index, total_stops):
        """Generate realistic GPS coordinates for bus stops"""
        base_lat = self.city_center['lat']
        base_lng = self.city_center['lng']
        
        # Create variation based on route and stop
        route_offset = (route_id - 500) * 0.01
        stop_offset = (stop_index / total_stops) * 0.02
        
        # Add some randomness for realistic coordinates
        lat_variation = np.random.normal(0, 0.005)
        lng_variation = np.random.normal(0, 0.005)
        
        lat = base_lat + route_offset + stop_offset + lat_variation
        lng = base_lng + route_offset + stop_offset + lng_variation
        
        return round(lat, 6), round(lng, 6)
    
    def generate_ridership_pattern(self, hour, day_of_week, weather_factor=1.0):
        """Generate realistic ridership patterns based on time and conditions"""
        base_ridership = 50
        
        # Hour-based patterns (rush hours have higher ridership)
        if 7 <= hour <= 9:  # Morning rush
            hourly_multiplier = 3.5
        elif 17 <= hour <= 19:  # Evening rush
            hourly_multiplier = 3.2
        elif 10 <= hour <= 16:  # Daytime
            hourly_multiplier = 2.0
        elif 20 <= hour <= 22:  # Evening
            hourly_multiplier = 1.5
        else:  # Night/early morning
            hourly_multiplier = 0.5
        
        # Day of week patterns (weekdays vs weekends)
        if day_of_week < 5:  # Weekdays
            day_multiplier = 1.2
        else:  # Weekends
            day_multiplier = 0.7
        
        # Calculate base ridership
        ridership = base_ridership * hourly_multiplier * day_multiplier * weather_factor
        
        # Add random variation
        ridership += np.random.normal(0, ridership * 0.15)
        
        return max(0, int(ridership))
    
    def generate_weather_factor(self, date):
        """Generate weather impact factor"""
        # Simulate seasonal and random weather effects
        day_of_year = date.timetuple().tm_yday
        
        # Seasonal pattern (monsoon season has lower ridership)
        seasonal_factor = 1.0
        if 150 < day_of_year < 250:  # Monsoon season
            seasonal_factor = 0.8
        
        # Random weather events
        if random.random() < 0.1:  # 10% chance of bad weather
            weather_factor = random.uniform(0.5, 0.8)
        else:
            weather_factor = random.uniform(0.9, 1.1)
        
        return seasonal_factor * weather_factor
    
    def generate_gps_logs(self):
        """Generate GPS tracking logs for buses"""
        gps_logs = []
        
        current_time = self.start_date
        
        while current_time < self.end_date:
            for route_id, route_info in self.bangalore_routes.items():
                stops = route_info['stops']
                
                # Generate multiple bus trips per hour for each route
                trips_per_hour = random.randint(2, 6)
                
                for trip in range(trips_per_hour):
                    bus_id = f"KA-01-{route_id}-{random.randint(1000, 9999)}"
                    trip_start_time = current_time + timedelta(minutes=random.randint(0, 59))
                    
                    # Generate GPS points for each stop
                    for stop_idx, stop_name in enumerate(stops):
                        lat, lng = self.generate_gps_coordinates(route_id, stop_idx, len(stops))
                        
                        # Add travel time between stops
                        stop_time = trip_start_time + timedelta(minutes=stop_idx * 5)
                        
                        # Add some GPS noise and speed variation
                        speed = random.randint(15, 45)  # km/h
                        
                        gps_logs.append({
                            'timestamp': stop_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'bus_id': bus_id,
                            'route_id': route_id,
                            'lat': lat,
                            'lng': lng,
                            'speed': speed,
                            'stop_name': stop_name,
                            'stop_sequence': stop_idx + 1
                        })
            
            current_time += timedelta(hours=1)
        
        return pd.DataFrame(gps_logs)
    
    def generate_ridership_data(self):
        """Generate hourly ridership data"""
        ridership_data = []
        
        current_time = self.start_date
        
        while current_time < self.end_date:
            weather_factor = self.generate_weather_factor(current_time.date())
            
            for route_id, route_info in self.bangalore_routes.items():
                # Generate ridership for each hour
                base_ridership = self.generate_ridership_pattern(
                    current_time.hour,
                    current_time.weekday(),
                    weather_factor
                )
                
                # Add route-specific multipliers based on route popularity
                route_multipliers = {500: 1.5, 501: 1.3, 502: 1.8, 503: 1.1, 504: 1.4}
                ridership = int(base_ridership * route_multipliers.get(route_id, 1.0))
                
                # Calculate occupancy percentage
                bus_capacity = 60  # Typical city bus capacity
                occupancy = min(100, (ridership / bus_capacity) * 100)
                
                ridership_data.append({
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'route_id': route_id,
                    'ridership': ridership,
                    'occupancy_percent': round(occupancy, 1),
                    'weather_factor': round(weather_factor, 2),
                    'hour': current_time.hour,
                    'day_of_week': current_time.weekday(),
                    'is_weekend': current_time.weekday() >= 5,
                    'is_rush_hour': current_time.hour in [7, 8, 9, 17, 18, 19]
                })
            
            current_time += timedelta(hours=1)
        
        return pd.DataFrame(ridership_data)
    
    def generate_route_master_data(self):
        """Generate route master data"""
        route_data = []
        
        for route_id, route_info in self.bangalore_routes.items():
            route_data.append({
                'route_id': route_id,
                'route_name': f"Route {route_id}",
                'source': route_info['source'],
                'destination': route_info['destination'],
                'distance_km': route_info['distance_km'],
                'fare': route_info['fare'],
                'stops': json.dumps(route_info['stops']),
                'total_stops': len(route_info['stops']),
                'estimated_travel_time': len(route_info['stops']) * 5,  # 5 minutes per stop
                'operational_status': 'active'
            })
        
        return pd.DataFrame(route_data)
    
    def save_generated_data(self):
        """Generate and save all datasets"""
        print("Generating GPS logs...")
        gps_df = self.generate_gps_logs()
        gps_df.to_csv('/app/backend/data/gps_logs.csv', index=False)
        print(f"Generated {len(gps_df)} GPS log entries")
        
        print("Generating ridership data...")
        ridership_df = self.generate_ridership_data()
        ridership_df.to_json('/app/backend/data/ridership.json', orient='records', date_format='iso')
        print(f"Generated {len(ridership_df)} ridership entries")
        
        print("Generating route master data...")
        routes_df = self.generate_route_master_data()
        routes_df.to_csv('/app/backend/data/routes_master.csv', index=False)
        print(f"Generated {len(routes_df)} route entries")
        
        # Generate summary statistics
        summary = {
            'generation_date': datetime.now().isoformat(),
            'data_period': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'duration_days': (self.end_date - self.start_date).days
            },
            'statistics': {
                'total_gps_logs': len(gps_df),
                'total_ridership_entries': len(ridership_df),
                'total_routes': len(routes_df),
                'avg_daily_ridership': ridership_df.groupby(ridership_df['timestamp'].str[:10])['ridership'].sum().mean(),
                'peak_hour_ridership': ridership_df[ridership_df['is_rush_hour']]['ridership'].mean(),
                'off_peak_ridership': ridership_df[~ridership_df['is_rush_hour']]['ridership'].mean()
            }
        }
        
        with open('/app/backend/data/data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nData generation completed successfully!")
        print(f"Files saved in /app/backend/data/")
        print(f"Summary: {summary['statistics']}")

if __name__ == "__main__":
    # Create data directory
    import os
    os.makedirs('/app/backend/data', exist_ok=True)
    
    # Generate data
    generator = BangaloreBusDataGenerator()
    generator.save_generated_data()