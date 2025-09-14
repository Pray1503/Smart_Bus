#!/usr/bin/env python3
"""
Smart Bus Management System - Route Optimization
Implements schedule optimization using rule-based algorithms and PuLP linear programming
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Try to import PuLP for linear programming optimization
try:
    from pulp import *
    HAS_PULP = True
except ImportError:
    print("PuLP not available, using rule-based optimization only")
    HAS_PULP = False

class BusScheduleOptimizer:
    """Optimize bus schedules to minimize wait times and improve efficiency"""
    
    def __init__(self):
        self.routes_data = {}
        self.ridership_patterns = {}
        self.optimization_results = {}
        
    def load_data(self, data_dir='/app/backend/data'):
        """Load necessary data for optimization"""
        data_path = Path(data_dir)
        
        # Load route master data
        routes_file = data_path / 'routes_master.csv'
        if routes_file.exists():
            routes_df = pd.read_csv(routes_file)
            self.routes_data = routes_df.set_index('route_id').to_dict('index')
        
        # Load ridership patterns
        ridership_file = data_path / 'ridership_clean.csv'
        if ridership_file.exists():
            ridership_df = pd.read_csv(ridership_file)
            ridership_df['timestamp'] = pd.to_datetime(ridership_df['timestamp'])
            ridership_df['hour'] = ridership_df['timestamp'].dt.hour
            
            # Create hourly patterns by route
            hourly_patterns = ridership_df.groupby(['route_id', 'hour']).agg({
                'ridership': 'mean',
                'occupancy_percent': 'mean'
            }).reset_index()
            
            for route_id in hourly_patterns['route_id'].unique():
                route_pattern = hourly_patterns[hourly_patterns['route_id'] == route_id]
                self.ridership_patterns[route_id] = route_pattern.set_index('hour').to_dict('index')
        
        print(f"Loaded data for {len(self.routes_data)} routes")
    
    def calculate_current_wait_times(self, route_id: int, current_schedule: Dict[str, List]) -> Dict[int, float]:
        """Calculate current average wait times by hour"""
        wait_times = {}
        
        if route_id not in self.ridership_patterns:
            return wait_times
        
        for hour in range(24):
            # Get ridership for this hour
            hourly_data = self.ridership_patterns[route_id].get(hour, {})
            ridership = hourly_data.get('ridership', 0)
            
            # Current frequency (buses per hour)
            current_frequency = current_schedule.get('frequencies', {}).get(str(hour), 2)  # Default 2 buses/hour
            
            if current_frequency > 0 and ridership > 0:
                # Simple wait time calculation: half the headway
                headway_minutes = 60 / current_frequency
                avg_wait_time = headway_minutes / 2
                
                # Adjust for overcrowding (higher wait times when buses are full)
                occupancy = hourly_data.get('occupancy_percent', 50)
                if occupancy > 80:
                    avg_wait_time *= 1.5  # 50% longer wait when overcrowded
                elif occupancy > 60:
                    avg_wait_time *= 1.2  # 20% longer wait
                
                wait_times[hour] = avg_wait_time
            else:
                wait_times[hour] = 30.0  # Default high wait time for no service
        
        return wait_times
    
    def rule_based_optimization(self, route_id: int) -> Dict:
        """Rule-based schedule optimization"""
        print(f"Applying rule-based optimization for Route {route_id}")
        
        if route_id not in self.ridership_patterns:
            return {"error": "No ridership data available"}
        
        # Current schedule (baseline)
        current_schedule = {
            'frequencies': {str(h): 2 for h in range(6, 23)},  # 2 buses/hour during service
            'service_hours': list(range(6, 23))
        }
        
        # Calculate current metrics
        current_wait_times = self.calculate_current_wait_times(route_id, current_schedule)
        current_avg_wait = np.mean(list(current_wait_times.values()))
        
        # Optimized schedule
        optimized_frequencies = {}
        
        for hour in range(24):
            hourly_data = self.ridership_patterns[route_id].get(hour, {})
            ridership = hourly_data.get('ridership', 0)
            occupancy = hourly_data.get('occupancy_percent', 0)
            
            if ridership == 0 or hour < 5 or hour > 23:
                # No service during very late/early hours
                optimized_frequencies[hour] = 0
            elif hour in [7, 8, 17, 18, 19]:  # Rush hours
                # High frequency during rush hours
                if ridership > 80:
                    optimized_frequencies[hour] = 6  # Every 10 minutes
                elif ridership > 50:
                    optimized_frequencies[hour] = 4  # Every 15 minutes
                else:
                    optimized_frequencies[hour] = 3  # Every 20 minutes
            elif hour in [9, 10, 11, 12, 13, 14, 15, 16]:  # Daytime
                # Moderate frequency
                if ridership > 60:
                    optimized_frequencies[hour] = 3
                elif ridership > 30:
                    optimized_frequencies[hour] = 2
                else:
                    optimized_frequencies[hour] = 1
            else:  # Evening/night
                # Lower frequency
                if ridership > 40:
                    optimized_frequencies[hour] = 2
                elif ridership > 20:
                    optimized_frequencies[hour] = 1
                else:
                    optimized_frequencies[hour] = 0
        
        # Anti-bunching rules
        optimized_schedule = self._apply_anti_bunching_rules(route_id, optimized_frequencies)
        
        # Calculate optimized metrics
        optimized_wait_times = self.calculate_current_wait_times(route_id, {'frequencies': optimized_schedule})
        optimized_avg_wait = np.mean(list(optimized_wait_times.values()))
        
        # Calculate fuel savings (fewer buses during low demand)
        current_total_buses = sum(current_schedule['frequencies'].values())
        optimized_total_buses = sum(optimized_schedule.values())
        fuel_savings_pct = max(0, (current_total_buses - optimized_total_buses) / current_total_buses * 100)
        
        return {
            'route_id': route_id,
            'optimization_type': 'rule_based',
            'current_schedule': current_schedule,
            'optimized_schedule': optimized_schedule,
            'metrics': {
                'current_avg_wait_time': round(current_avg_wait, 2),
                'optimized_avg_wait_time': round(optimized_avg_wait, 2),
                'wait_time_reduction_pct': round((current_avg_wait - optimized_avg_wait) / current_avg_wait * 100, 1),
                'fuel_savings_pct': round(fuel_savings_pct, 1),
                'current_total_buses': current_total_buses,
                'optimized_total_buses': optimized_total_buses
            }
        }
    
    def _apply_anti_bunching_rules(self, route_id: int, frequencies: Dict[int, int]) -> Dict[int, int]:
        """Apply anti-bunching rules to prevent buses from clustering"""
        optimized = frequencies.copy()
        
        route_info = self.routes_data.get(route_id, {})
        travel_time_minutes = route_info.get('estimated_travel_time', 30)
        
        for hour in range(24):
            current_freq = optimized.get(hour, 0)
            
            if current_freq > 0:
                # Ensure minimum headway to prevent bunching
                min_headway = max(10, travel_time_minutes / current_freq * 0.3)  # At least 30% of travel time
                max_frequency = 60 / min_headway
                
                if current_freq > max_frequency:
                    optimized[hour] = int(max_frequency)
                
                # Check adjacent hours for consistency
                prev_freq = optimized.get(hour - 1, 0)
                next_freq = optimized.get(hour + 1, 0)
                
                # Smooth transitions between hours
                if prev_freq > 0 and abs(current_freq - prev_freq) > 2:
                    optimized[hour] = min(current_freq, prev_freq + 1)
        
        return optimized
    
    def linear_programming_optimization(self, route_id: int, forecast_data: Optional[pd.DataFrame] = None) -> Dict:
        """Advanced optimization using linear programming"""
        if not HAS_PULP:
            return self.rule_based_optimization(route_id)
        
        print(f"Applying LP optimization for Route {route_id}")
        
        if route_id not in self.ridership_patterns:
            return {"error": "No ridership data available"}
        
        # Create LP problem
        prob = LpProblem(f"BusScheduleOptimization_Route_{route_id}", LpMinimize)
        
        # Decision variables: number of buses per hour
        hours = range(5, 24)  # Service hours
        buses = LpVariable.dicts("buses", hours, lowBound=0, upBound=8, cat='Integer')
        
        # Objective function: minimize total cost (wait time cost + operational cost)
        wait_time_cost = 0
        operational_cost = 0
        
        for hour in hours:
            hourly_data = self.ridership_patterns[route_id].get(hour, {})
            ridership = hourly_data.get('ridership', 0)
            
            if ridership > 0:
                # Wait time cost (inversely proportional to frequency)
                wait_time_cost += ridership * (30 / (buses[hour] + 0.1))  # Add small value to avoid division by zero
                
                # Operational cost (proportional to number of buses)
                operational_cost += buses[hour] * 100  # Cost per bus per hour
        
        prob += wait_time_cost + operational_cost
        
        # Constraints
        for hour in hours:
            hourly_data = self.ridership_patterns[route_id].get(hour, {})
            ridership = hourly_data.get('ridership', 0)
            
            # Minimum service level
            if ridership > 20:
                prob += buses[hour] >= 1
            
            # Maximum capacity constraint
            bus_capacity = 60
            prob += buses[hour] * bus_capacity >= ridership * 0.8  # 80% of demand covered
            
            # Rush hour minimum service
            if hour in [7, 8, 17, 18, 19] and ridership > 30:
                prob += buses[hour] >= 2
        
        # Fleet size constraint (total buses available)
        max_fleet_size = 10
        for hour in hours:
            prob += buses[hour] <= max_fleet_size
        
        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=0))
        
        if prob.status != 1:  # Not optimal
            print(f"LP optimization failed for Route {route_id}, falling back to rule-based")
            return self.rule_based_optimization(route_id)
        
        # Extract solution
        optimized_schedule = {}
        for hour in range(24):
            if hour in hours:
                optimized_schedule[hour] = int(buses[hour].varValue or 0)
            else:
                optimized_schedule[hour] = 0
        
        # Calculate metrics
        current_schedule = {'frequencies': {str(h): 2 for h in range(6, 23)}}
        current_wait_times = self.calculate_current_wait_times(route_id, current_schedule)
        optimized_wait_times = self.calculate_current_wait_times(route_id, {'frequencies': optimized_schedule})
        
        current_avg_wait = np.mean(list(current_wait_times.values()))
        optimized_avg_wait = np.mean(list(optimized_wait_times.values()))
        
        return {
            'route_id': route_id,
            'optimization_type': 'linear_programming',
            'optimized_schedule': optimized_schedule,
            'metrics': {
                'current_avg_wait_time': round(current_avg_wait, 2),
                'optimized_avg_wait_time': round(optimized_avg_wait, 2),
                'wait_time_reduction_pct': round((current_avg_wait - optimized_avg_wait) / current_avg_wait * 100, 1),
                'objective_value': prob.objective.value()
            }
        }
    
    def optimize_all_routes(self, method='rule_based') -> Dict[int, Dict]:
        """Optimize schedules for all routes"""
        results = {}
        
        for route_id in self.routes_data.keys():
            try:
                if method == 'linear_programming':
                    result = self.linear_programming_optimization(route_id)
                else:
                    result = self.rule_based_optimization(route_id)
                
                results[route_id] = result
                print(f"Route {route_id}: {result['metrics'].get('wait_time_reduction_pct', 0):.1f}% wait time reduction")
                
            except Exception as e:
                print(f"Optimization failed for Route {route_id}: {str(e)}")
                results[route_id] = {'error': str(e)}
        
        self.optimization_results = results
        return results
    
    def generate_optimization_summary(self) -> Dict:
        """Generate summary of optimization results"""
        if not self.optimization_results:
            return {}
        
        successful_optimizations = [
            r for r in self.optimization_results.values() 
            if 'error' not in r and 'metrics' in r
        ]
        
        if not successful_optimizations:
            return {"error": "No successful optimizations"}
        
        total_wait_reduction = np.mean([
            r['metrics']['wait_time_reduction_pct'] 
            for r in successful_optimizations
        ])
        
        total_fuel_savings = np.mean([
            r['metrics'].get('fuel_savings_pct', 0) 
            for r in successful_optimizations
        ])
        
        return {
            'optimization_summary': {
                'routes_optimized': len(successful_optimizations),
                'avg_wait_time_reduction_pct': round(total_wait_reduction, 1),
                'avg_fuel_savings_pct': round(total_fuel_savings, 1),
                'optimization_date': datetime.now().isoformat()
            },
            'route_details': self.optimization_results
        }
    
    def save_optimization_results(self, output_path='/app/backend/data/optimization_results.json'):
        """Save optimization results to file"""
        summary = self.generate_optimization_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Optimization results saved to {output_path}")
        return summary

def main():
    """Main function to run optimization"""
    print("Starting bus schedule optimization...")
    
    optimizer = BusScheduleOptimizer()
    
    # Load data
    optimizer.load_data()
    
    # Run optimization
    results = optimizer.optimize_all_routes(method='rule_based')
    
    # Generate and save summary
    summary = optimizer.save_optimization_results()
    
    print("\nOptimization Summary:")
    if 'optimization_summary' in summary:
        opt_summary = summary['optimization_summary']
        print(f"Routes optimized: {opt_summary['routes_optimized']}")
        print(f"Average wait time reduction: {opt_summary['avg_wait_time_reduction_pct']}%")
        print(f"Average fuel savings: {opt_summary['avg_fuel_savings_pct']}%")
    
    print("Optimization completed successfully!")

if __name__ == "__main__":
    main()