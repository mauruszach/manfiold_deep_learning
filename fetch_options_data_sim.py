#!/usr/bin/env python3
"""
Options Chain Data Simulator for Manifold Learning
-------------------------------------------------
This script simulates options chain data for manifold learning
when Interactive Brokers market data subscription is not available.
"""

import numpy as np
import pandas as pd
import datetime as dt
import time
import json
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import math
import random

# Configure command line arguments
parser = argparse.ArgumentParser(description='Simulate options data for manifold learning')
parser.add_argument('--symbol', type=str, default='SPY', help='Ticker symbol (default: SPY)')
parser.add_argument('--days', type=int, default=30, help='Days to expiration to focus on (default: 30)')
parser.add_argument('--interval', type=int, default=600, help='Time between data fetches in seconds (default: 600)')
parser.add_argument('--samples', type=int, default=100, help='Number of samples to collect (default: 100)')
parser.add_argument('--price', type=float, default=450.0, help='Current price of the underlying (default: 450.0 for SPY)')
parser.add_argument('--volatility', type=float, default=0.2, help='Base implied volatility (default: 0.20)')
args = parser.parse_args()

class OptionsDataSimulator:
    def __init__(self, symbol='SPY', target_days=30, current_price=None, base_volatility=None):
        """
        Initialize the options data simulator.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol to simulate options for
        target_days : int
            Target days to expiration to focus on
        current_price : float
            Current price of the underlying (if None, uses defaults based on symbol)
        base_volatility : float
            Base implied volatility to use for simulations (if None, uses defaults)
        """
        self.symbol = symbol
        self.target_days = target_days
        
        # Set default prices based on symbol if not provided
        if current_price is None:
            if symbol == 'SPY':
                self.current_price = 450.0
            elif symbol == 'AAPL':
                self.current_price = 180.0
            elif symbol == 'MSFT':
                self.current_price = 400.0
            elif symbol == 'GOOGL':
                self.current_price = 140.0
            elif symbol == 'AMZN':
                self.current_price = 175.0
            else:
                self.current_price = 100.0  # Default for unknown symbols
        else:
            self.current_price = current_price
            
        # Set base volatility if not provided
        self.base_volatility = 0.2 if base_volatility is None else base_volatility
        
        # Create data directory
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'options_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Data storage
        self.all_data = []
        self.samples_collected = 0
        self.metadata = {
            'symbol': symbol,
            'target_days': target_days,
            'first_collection': None,
            'last_collection': None,
            'total_samples': 0
        }
        
        # Initialize price movement simulation
        self.price_volatility = 0.01  # 1% daily volatility
        self.trend = random.uniform(-0.0005, 0.0005)  # Very small trend component
        
        print(f"Initialized options simulator for {symbol} at ${self.current_price:.2f} with {target_days} DTE")
        
    def black_scholes_d1(self, S, K, T, r, sigma):
        """Calculate d1 term in Black-Scholes formula"""
        return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def black_scholes_d2(self, d1, sigma, T):
        """Calculate d2 term in Black-Scholes formula"""
        return d1 - sigma * np.sqrt(T)
    
    def bs_call_price(self, S, K, T, r, sigma):
        """Calculate call option price using Black-Scholes model"""
        d1 = self.black_scholes_d1(S, K, T, r, sigma)
        d2 = self.black_scholes_d2(d1, sigma, T)
        return S * self.norm_cdf(d1) - K * np.exp(-r * T) * self.norm_cdf(d2)
    
    def bs_put_price(self, S, K, T, r, sigma):
        """Calculate put option price using Black-Scholes model"""
        d1 = self.black_scholes_d1(S, K, T, r, sigma)
        d2 = self.black_scholes_d2(d1, sigma, T)
        return K * np.exp(-r * T) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)
    
    def norm_cdf(self, x):
        """Standard normal cumulative distribution function"""
        return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0
    
    def norm_pdf(self, x):
        """Standard normal probability density function"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        
    def calculate_delta(self, S, K, T, r, sigma, option_type):
        """Calculate option delta"""
        d1 = self.black_scholes_d1(S, K, T, r, sigma)
        if option_type == 'C':
            return self.norm_cdf(d1)
        else:  # Put
            return self.norm_cdf(d1) - 1
            
    def calculate_gamma(self, S, K, T, r, sigma):
        """Calculate option gamma (same for calls and puts)"""
        d1 = self.black_scholes_d1(S, K, T, r, sigma)
        return self.norm_pdf(d1) / (S * sigma * np.sqrt(T))
        
    def calculate_theta(self, S, K, T, r, sigma, option_type):
        """Calculate option theta (time decay)"""
        d1 = self.black_scholes_d1(S, K, T, r, sigma)
        d2 = self.black_scholes_d2(d1, sigma, T)
        
        term1 = -S * self.norm_pdf(d1) * sigma / (2 * np.sqrt(T))
        
        if option_type == 'C':
            term2 = -r * K * np.exp(-r * T) * self.norm_cdf(d2)
            return (term1 + term2) / 365  # Convert to daily theta
        else:  # Put
            term2 = r * K * np.exp(-r * T) * self.norm_cdf(-d2)
            return (term1 + term2) / 365  # Convert to daily theta
            
    def calculate_vega(self, S, K, T, r, sigma):
        """Calculate option vega (sensitivity to volatility)"""
        d1 = self.black_scholes_d1(S, K, T, r, sigma)
        return S * np.sqrt(T) * self.norm_pdf(d1) / 100  # Convert to 1% move
    
    def simulate_volatility_smile(self, moneyness, days_to_expiry):
        """Simulate volatility smile based on moneyness and days to expiry"""
        # Create realistic volatility smile (higher on wings)
        atm_vol = self.base_volatility
        
        # Adjust volatility based on days to expiry (term structure)
        if days_to_expiry < 14:
            # Short term options have steeper smile
            term_factor = 1.2
        elif days_to_expiry < 45:
            # Medium term options have moderate smile
            term_factor = 1.0
        else:
            # Long term options have flatter smile
            term_factor = 0.8
            
        # Create smile pattern - higher at wings, lowest ATM
        wing_factor = 0.15 * term_factor * (moneyness - 1.0)**2
        
        # Add slight skew (puts > calls) common in equity markets
        skew_factor = -0.02 * (moneyness - 1.0) * term_factor
        
        # Add randomness to make it more realistic
        random_factor = random.uniform(-0.01, 0.01)
        
        # Combine all factors
        implied_vol = max(0.05, atm_vol + wing_factor + skew_factor + random_factor)
        
        return implied_vol
    
    def update_underlying_price(self):
        """Update the underlying price with realistic random movements"""
        # Random component (daily volatility scaled to update interval)
        random_component = np.random.normal(0, self.price_volatility / np.sqrt(252))
        
        # Apply a very small trend and random component
        self.current_price *= (1 + self.trend + random_component)
        
        # Occasionally flip the trend direction
        if random.random() < 0.05:  # 5% chance of trend change
            self.trend *= -1
            
        return self.current_price
    
    def generate_expiry_date(self, target_days):
        """Generate a realistic options expiration date based on target days"""
        # Find the closest Friday
        target_date = dt.date.today() + dt.timedelta(days=target_days)
        days_to_friday = (4 - target_date.weekday()) % 7
        expiry_date = target_date + dt.timedelta(days=days_to_friday)
        
        # If we're really close to expiration, go to next Friday
        if days_to_friday == 0 and target_days < 7:
            expiry_date += dt.timedelta(days=7)
            
        # Format as YYYYMMDD
        expiry_string = expiry_date.strftime('%Y%m%d')
        return expiry_string, (expiry_date - dt.date.today()).days
    
    def simulate_options_chain(self):
        """Simulate a realistic options chain for the given parameters"""
        # Update the underlying price
        underlying_price = self.update_underlying_price()
        
        # Find expiration date closest to target days
        expiry_string, days_to_expiry = self.generate_expiry_date(self.target_days)
        
        # Generate a range of strike prices (±20% around current price)
        min_strike = underlying_price * 0.8
        max_strike = underlying_price * 1.2
        
        # Create more strikes around the current price, fewer at the edges
        n_strikes = 15  # Number of strikes
        
        # Non-linear strike spacing - more dense near the money
        moneyness_range = np.linspace(-0.2, 0.2, n_strikes)
        strikes = [underlying_price * (1 + m) for m in moneyness_range]
        
        # Round strikes to appropriate precision
        if underlying_price < 50:
            strikes = [round(s * 2) / 2 for s in strikes]  # 0.5 increments
        elif underlying_price < 100:
            strikes = [round(s) for s in strikes]  # 1.0 increments
        else:
            strikes = [round(s / 5) * 5 for s in strikes]  # 5.0 increments
            
        # Remove duplicates and sort
        strikes = sorted(list(set(strikes)))
        
        # Set risk-free rate
        risk_free_rate = 0.04  # 4% - adjust as needed
        
        # Calculate time to expiry in years
        T = days_to_expiry / 365
        
        # Generate options data
        options_data = []
        
        # Simulate options at each strike for both calls and puts
        for strike in strikes:
            for right in ['C', 'P']:
                # Calculate moneyness
                moneyness = strike / underlying_price
                
                # Simulate implied volatility with smile effect
                implied_vol = self.simulate_volatility_smile(moneyness, days_to_expiry)
                
                # Calculate option price
                if right == 'C':
                    price = self.bs_call_price(underlying_price, strike, T, risk_free_rate, implied_vol)
                else:
                    price = self.bs_put_price(underlying_price, strike, T, risk_free_rate, implied_vol)
                
                # Calculate greeks
                delta = self.calculate_delta(underlying_price, strike, T, risk_free_rate, implied_vol, right)
                gamma = self.calculate_gamma(underlying_price, strike, T, risk_free_rate, implied_vol)
                theta = self.calculate_theta(underlying_price, strike, T, risk_free_rate, implied_vol, right)
                vega = self.calculate_vega(underlying_price, strike, T, risk_free_rate, implied_vol)
                
                # Simulate volume and open interest (higher near ATM)
                atm_factor = np.exp(-10 * (np.log(moneyness))**2)
                base_volume = int(1000 * atm_factor)
                volume = max(1, int(base_volume * (0.8 + 0.4 * random.random())))
                open_interest = max(10, int(volume * 5 * (0.8 + 0.4 * random.random())))
                
                # Create option data
                option_data = {
                    'symbol': self.symbol,
                    'expiry': expiry_string,
                    'strike': strike,
                    'right': right,
                    'days_to_expiry': days_to_expiry,
                    'price': price,
                    'bid': price * 0.98,  # Simulate bid-ask spread
                    'ask': price * 1.02,
                    'volume': volume,
                    'open_interest': open_interest,
                    'implied_volatility': implied_vol,
                    'delta': delta,
                    'gamma': gamma,
                    'vega': vega,
                    'theta': theta,
                    'underlying_price': underlying_price,
                    'timestamp': dt.datetime.now().isoformat()
                }
                
                options_data.append(option_data)
        
        print(f"Simulated {len(options_data)} options contracts at underlying price ${underlying_price:.2f}")
        
        return options_data
    
    def save_data(self, options_data):
        """Save the options data to CSV and update the dataset for manifold learning"""
        if not options_data:
            return False
        
        # Save raw data snapshot
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(self.data_dir, f"{self.symbol}_options_{timestamp}.csv")
        
        df = pd.DataFrame(options_data)
        df.to_csv(csv_path, index=False)
        
        # Append to all data
        self.all_data.extend(options_data)
        
        # Save metadata
        metadata_path = os.path.join(self.data_dir, f"{self.symbol}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Create a dataset suitable for manifold learning
        self.prepare_manifold_dataset()
        
        return True
    
    def prepare_manifold_dataset(self):
        """Prepare the options data for manifold learning"""
        if not self.all_data:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_data)
        
        # Select key features for the manifold
        features = ['strike', 'days_to_expiry', 'price', 'implied_volatility', 
                   'delta', 'gamma', 'vega', 'theta', 'underlying_price']
        
        # Add call/put as a numeric feature (1 for call, -1 for put)
        df['call_put'] = df['right'].apply(lambda x: 1 if x == 'C' else -1)
        features.append('call_put')
        
        # Filter to available data
        df_filtered = df[features].copy()
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_filtered)
        
        # Save both raw and scaled data for manifold learning
        raw_path = os.path.join(self.data_dir, f"{self.symbol}_manifold_raw.csv")
        scaled_path = os.path.join(self.data_dir, f"{self.symbol}_manifold_scaled.csv")
        
        df_filtered.to_csv(raw_path, index=False)
        pd.DataFrame(scaled_data, columns=df_filtered.columns).to_csv(scaled_path, index=False)
        
        # Create a metric tensor from the data to feed into our manifold learning system
        self.generate_metric_tensor(scaled_data)
        
        print(f"Created manifold dataset with {len(df_filtered)} samples and {len(features)} features")
        return df_filtered
    
    def generate_metric_tensor(self, data):
        """Generate a metric tensor from the options data for manifold visualization"""
        if len(data) < 10:  # Need sufficient data
            return
        
        # Convert data to numpy array if it's not already
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Compute covariance matrix as a simple metric tensor
        # In a more sophisticated approach, this could be a learned metric
        cov_matrix = np.cov(data, rowvar=False)
        
        # Ensure it's positive definite for a proper Riemannian metric
        # Add a small value to the diagonal if needed
        min_eig = np.min(np.linalg.eigvals(cov_matrix))
        if min_eig < 1e-6:
            cov_matrix += np.eye(cov_matrix.shape[0]) * (1e-6 - min_eig)
        
        # Save the metric tensor to the appropriate locations for visualization
        paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metric_tensor.csv'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paths_on_manifold', 'metric_tensor.csv')
        ]
        
        for path in paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savetxt(path, cov_matrix, delimiter=',')
        
        # Generate simplified Christoffel symbols from the metric
        dim = cov_matrix.shape[0]
        christoffel = np.zeros((dim, dim, dim))
        
        # For each component of the Christoffel symbols
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    # Use a simple approximation (normally would calculate this properly)
                    # In a constant metric space, these would be zero
                    # Here we generate small values for visualization
                    christoffel[i, j, k] = 0.01 * (cov_matrix[i, j] * cov_matrix[i, k] / (1 + cov_matrix[i, i]))
        
        # Save Christoffel symbols
        christo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'christoffel_symbols.csv')
        with open(christo_path, 'w') as f:
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        value = christoffel[i, j, k]
                        if abs(value) > 1e-10:  # Only save non-zero values
                            f.write(f"{i},{j},{k},{value}\n")
        
        # Update metadata for the visualization
        metadata = {
            'epoch': self.samples_collected,
            'total_epochs': args.samples,
            'step': self.samples_collected,
            'total_steps': args.samples,
            'loss': 0.01,  # Placeholder
            'is_training': True
        }
        
        metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metric_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Generated {dim}x{dim} metric tensor from options data")
    
    def visualize_data(self):
        """Create a simple visualization of the options data"""
        if not self.all_data:
            return
        
        df = pd.DataFrame(self.all_data)
        
        # Filter to the most recent snapshot
        timestamp = df['timestamp'].max()
        recent_df = df[df['timestamp'] == timestamp]
        
        # Create a volatility smile plot
        plt.figure(figsize=(12, 8))
        
        # Plot calls and puts separately
        calls = recent_df[recent_df['right'] == 'C']
        puts = recent_df[recent_df['right'] == 'P']
        
        if not calls.empty:
            plt.scatter(calls['strike'], calls['implied_volatility'], 
                      label='Calls', color='green', alpha=0.7)
        
        if not puts.empty:
            plt.scatter(puts['strike'], puts['implied_volatility'], 
                      label='Puts', color='red', alpha=0.7)
        
        # Add vertical line for current underlying price
        if 'underlying_price' in recent_df.columns and not recent_df.empty:
            underlying_price = recent_df['underlying_price'].iloc[0]
            plt.axvline(x=underlying_price, color='blue', linestyle='--', 
                      label=f'Current Price: ${underlying_price:.2f}')
        
        plt.title(f'{self.symbol} Implied Volatility Smile (Simulated)')
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        viz_path = os.path.join(self.data_dir, f"{self.symbol}_vol_smile.png")
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Created volatility smile visualization: {viz_path}")
    
    def run_simulation(self, interval=600, samples=100):
        """Run the simulation process at specified intervals"""
        try:
            print(f"Starting options data simulation for {self.symbol} at ${self.current_price:.2f}")
            print(f"Will generate {samples} samples at {interval} second intervals")
            
            self.samples_collected = 0
            
            # Update metadata
            now = dt.datetime.now()
            self.metadata['first_collection'] = now.isoformat()
            
            while self.samples_collected < samples:
                print(f"\nSimulation {self.samples_collected + 1}/{samples} at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Generate and save simulated data
                options_data = self.simulate_options_chain()
                if options_data:
                    self.save_data(options_data)
                    self.visualize_data()
                    self.samples_collected += 1
                    
                    # Update metadata
                    self.metadata['last_collection'] = dt.datetime.now().isoformat()
                    self.metadata['total_samples'] = self.samples_collected
                else:
                    print("Failed to generate data in this iteration")
                
                # Wait for the next interval if not the last sample
                if self.samples_collected < samples:
                    print(f"Waiting {interval} seconds for next simulation...")
                    for i in range(interval, 0, -60):
                        time.sleep(min(60, i))
                        remaining = max(0, i - 60)
                        if remaining > 0:
                            print(f"{remaining} seconds remaining until next simulation...", end='\r')
                    print("")  # Clear line
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"Error during simulation: {e}")
        finally:
            print(f"Simulation complete. Generated {self.samples_collected} samples.")

if __name__ == "__main__":
    print(f"Starting options data simulation for {args.symbol}")
    
    simulator = OptionsDataSimulator(
        symbol=args.symbol,
        target_days=args.days,
        current_price=args.price,
        base_volatility=args.volatility
    )
    
    # Run the simulation process
    simulator.run_simulation(
        interval=args.interval,
        samples=args.samples
    )
    
    print("Options data simulation complete. Data saved to options_data directory.") 