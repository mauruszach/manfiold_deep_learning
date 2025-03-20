#!/usr/bin/env python3
"""
Options Chain Data Collector for Manifold Learning
-------------------------------------------------
This script connects to Interactive Brokers API to fetch options chain data
and structures it for use in manifold learning.
"""

import numpy as np
import pandas as pd
import datetime as dt
import time
import json
import os
from ib_insync import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse

# Configure command line arguments
parser = argparse.ArgumentParser(description='Fetch options data from Interactive Brokers')
parser.add_argument('--symbol', type=str, default='SPY', help='Ticker symbol (default: SPY)')
parser.add_argument('--days', type=int, default=30, help='Days to expiration to focus on (default: 30)')
parser.add_argument('--port', type=int, default=4001, help='IB Gateway/TWS port (default: 4001)')
parser.add_argument('--client_id', type=int, default=1, help='Client ID (default: 1)')
parser.add_argument('--interval', type=int, default=600, help='Time between data fetches in seconds (default: 600)')
parser.add_argument('--samples', type=int, default=100, help='Number of samples to collect (default: 100)')
args = parser.parse_args()

class OptionsDataCollector:
    def __init__(self, symbol='SPY', target_days=30, port=4001, client_id=1):
        """
        Initialize the options data collector.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol to fetch options for
        target_days : int
            Target days to expiration to focus on
        port : int
            IB Gateway/TWS port (4001 for API connection)
        client_id : int
            Client ID for TWS connection
        """
        self.symbol = symbol
        self.target_days = target_days
        self.port = port
        self.client_id = client_id
        
        # Create data directory
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'options_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize connection
        self.ib = IB()
        self.samples_collected = 0
        
        # Data storage
        self.all_data = []
        self.metadata = {
            'symbol': symbol,
            'target_days': target_days,
            'first_collection': None,
            'last_collection': None,
            'total_samples': 0
        }
        
    def connect(self):
        """Connect to Interactive Brokers TWS/Gateway"""
        try:
            if not self.ib.isConnected():
                self.ib.connect('127.0.0.1', self.port, clientId=self.client_id)
                print(f"Connected to IB (port {self.port}, client ID {self.client_id})")
            return True
        except Exception as e:
            print(f"Error connecting to IB: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IB"""
        if self.ib.isConnected():
            self.ib.disconnect()
            print("Disconnected from IB")
    
    def get_contract(self):
        """Get the stock contract for the symbol"""
        contract = Stock(self.symbol, 'SMART', 'USD')
        return contract
    
    def get_nearest_expiry(self, days):
        """Get the expiration date closest to the target days"""
        contract = self.get_contract()
        
        # Qualify the contract
        self.ib.qualifyContracts(contract)
        
        # Get available expiry dates
        expirations = self.ib.reqContractDetails(contract)[0].contract.secType
        if expirations:
            # For SPX, expirations typically include weeklies, monthlies, and quarterlies
            target_date = dt.date.today() + dt.timedelta(days=days)
            
            # Find closest expiry
            closest_expiry = None
            min_diff = float('inf')
            
            for expiry in expirations:
                try:
                    expiry_date = dt.datetime.strptime(expiry, '%Y%m%d').date()
                    diff = abs((expiry_date - target_date).days)
                    if diff < min_diff:
                        min_diff = diff
                        closest_expiry = expiry
                except ValueError:
                    continue
            
            return closest_expiry
        
        # If we couldn't get expirations, return a date approximately N days from now
        return (dt.date.today() + dt.timedelta(days=days)).strftime('%Y%m%d')
    
    def fetch_options_chain(self):
        """Fetch the full options chain for the symbol"""
        if not self.connect():
            return None
        
        contract = self.get_contract()
        
        # Get the current stock price
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(1)  # Wait for data to arrive
        current_price = ticker.marketPrice()
        self.ib.cancelMktData(contract)
        
        if not current_price or current_price <= 0:
            print("Could not get current price, using approximation")
            current_price = 100  # Approximation for testing
        
        # Get chains
        try:
            chains = self.ib.reqSecDefOptParams(
                self.symbol, '', 'STK', contract.conId
            )
            
            # Find nearest expiration to target days
            target_expiry = self.get_nearest_expiry(self.target_days)
            print(f"Target expiration: {target_expiry}")
            
            # Create option contracts for the chain
            option_chain = []
            
            # Determine strike range (e.g., ±20% around current price)
            min_strike = current_price * 0.8
            max_strike = current_price * 1.2
            
            # Create a list of strikes to query
            strikes = []
            for chain in chains:
                valid_strikes = [s for s in chain.strikes if min_strike <= s <= max_strike]
                strikes.extend(valid_strikes)
            
            # Deduplicate and sort strikes
            strikes = sorted(list(set(strikes)))
            
            # Query both calls and puts
            for strike in strikes:
                for right in ['C', 'P']:
                    option = Option(self.symbol, target_expiry, strike, right, 'SMART')
                    option_chain.append(option)
            
            # Qualify the contracts
            qualified_contracts = self.ib.qualifyContracts(*option_chain)
            
            # Request market data for all options
            tickers = []
            for contract in qualified_contracts:
                # Request delayed market data instead of real-time
                ticker = self.ib.reqMktData(contract, '', False, False)
                tickers.append(ticker)
            
            # Allow more time for delayed data to arrive
            self.ib.sleep(10)  # Increased from 5 to 10 seconds
            
            # Process the data
            options_data = []
            for ticker in tickers:
                contract = ticker.contract
                
                # Calculate days to expiration
                expiry_date = dt.datetime.strptime(contract.lastTradeDateOrContractMonth, '%Y%m%d').date()
                days_to_expiry = (expiry_date - dt.date.today()).days
                
                # Skip if no market price available
                if not ticker.marketPrice():
                    continue
                
                option_data = {
                    'symbol': self.symbol,
                    'expiry': contract.lastTradeDateOrContractMonth,
                    'strike': contract.strike,
                    'right': contract.right,
                    'days_to_expiry': days_to_expiry,
                    'price': ticker.marketPrice(),
                    'bid': ticker.bid,
                    'ask': ticker.ask,
                    'volume': ticker.volume,
                    'open_interest': ticker.openInterest,
                    'implied_volatility': ticker.impliedVolatility,
                    'delta': ticker.delta,
                    'gamma': ticker.gamma,
                    'vega': ticker.vega,
                    'theta': ticker.theta,
                    'underlying_price': current_price,
                    'timestamp': dt.datetime.now().isoformat()
                }
                
                options_data.append(option_data)
            
            # Cancel market data
            for ticker in tickers:
                self.ib.cancelMktData(ticker.contract)
            
            # Update metadata
            now = dt.datetime.now()
            if not self.metadata['first_collection']:
                self.metadata['first_collection'] = now.isoformat()
            self.metadata['last_collection'] = now.isoformat()
            self.metadata['total_samples'] += 1
            
            print(f"Collected {len(options_data)} options contracts data")
            
            return options_data
        
        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return None
        finally:
            # Don't disconnect here to reuse the connection for multiple fetches
            pass
    
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
        
        # Fill missing values
        df = df.dropna(subset=['price', 'strike', 'days_to_expiry'])
        
        # Select key features for the manifold
        features = ['strike', 'days_to_expiry', 'price', 'implied_volatility', 
                   'delta', 'gamma', 'vega', 'theta', 'underlying_price']
        
        # Filter to the features that are available (some Greeks might be missing)
        available_features = [f for f in features if f in df.columns]
        
        # Add call/put as a numeric feature (1 for call, -1 for put)
        df['call_put'] = df['right'].apply(lambda x: 1 if x == 'C' else -1)
        available_features.append('call_put')
        
        # Filter to available data
        df_filtered = df[available_features].copy()
        
        # Handle missing values
        df_filtered = df_filtered.fillna(0)
        
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
        
        print(f"Created manifold dataset with {len(df_filtered)} samples and {len(available_features)} features")
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
        
        plt.title(f'{self.symbol} Implied Volatility Smile')
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        viz_path = os.path.join(self.data_dir, f"{self.symbol}_vol_smile.png")
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Created volatility smile visualization: {viz_path}")
    
    def run_collection(self, interval=600, samples=100):
        """Run the data collection process at specified intervals"""
        self.connect()
        
        try:
            print(f"Starting options data collection for {self.symbol}, targeting {self.target_days} DTE")
            print(f"Will collect {samples} samples at {interval} second intervals")
            
            self.samples_collected = 0
            
            while self.samples_collected < samples:
                print(f"\nCollection {self.samples_collected + 1}/{samples} at {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Fetch and save data
                options_data = self.fetch_options_chain()
                
                if options_data:
                    self.save_data(options_data)
                    self.visualize_data()
                    self.samples_collected += 1
                else:
                    print("Failed to collect data in this iteration")
                
                # Wait for the next interval if not the last sample
                if self.samples_collected < samples:
                    print(f"Waiting {interval} seconds for next collection...")
                    for i in range(interval, 0, -60):
                        time.sleep(min(60, i))
                        remaining = max(0, i - 60)
                        if remaining > 0:
                            print(f"{remaining} seconds remaining until next collection...", end='\r')
                    print("")  # Clear line
        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        except Exception as e:
            print(f"Error during collection: {e}")
        finally:
            self.disconnect()
            print(f"Collection complete. Collected {self.samples_collected} samples.")

if __name__ == "__main__":
    print(f"Starting options data collection for {args.symbol}")
    
    collector = OptionsDataCollector(
        symbol=args.symbol,
        target_days=args.days,
        port=args.port,
        client_id=args.client_id
    )
    
    # Run the collection process
    collector.run_collection(
        interval=args.interval,
        samples=args.samples
    )
    
    print("Options data collection complete. Data saved to options_data directory.") 