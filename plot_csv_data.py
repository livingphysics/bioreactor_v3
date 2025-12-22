#!/usr/bin/env python3
"""
Standalone script to plot CSV data from bioreactor data files.

Reads a CSV file and plots data with automatic grouping of similar columns.
Groups columns by type (OD, Temperature) and updates the plot periodically.

Usage:
    python plot_csv_data.py <csv_file_path> [update_interval]
    
Example:
    python plot_csv_data.py bioreactor_data/20251210_134704_bioreactor_data.csv 5.0
"""

import csv
import os
import sys
import time
import threading
import matplotlib.pyplot as plt
import numpy as np


def plot_csv_data(csv_file_path: str, update_interval: float = 5.0):
    """
    Read CSV file and plot data with automatic grouping of similar columns.
    
    Groups columns by type:
    - OD readings (columns containing 'OD' or 'od') -> one subplot
    - Temperature (columns containing 'temp' or 'temperature') -> one subplot
    - Time -> x-axis for all
    
    Updates the plot periodically by re-reading the CSV file.
    
    Args:
        csv_file_path: Path to the CSV file to read
        update_interval: Time in seconds between plot updates (default: 5.0)
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found: {csv_file_path}")
        return
    
    # Global storage for plot data
    last_row_count = 0
    fig = None
    axes = None
    
    def group_columns(headers):
        """Group column headers by type."""
        groups = {
            'OD': [],
            'Temperature': [],
            'Time': []
        }
        
        for header in headers:
            header_lower = header.lower()
            if header_lower == 'time':
                groups['Time'].append(header)
            elif 'od' in header_lower:
                groups['OD'].append(header)
            elif 'temp' in header_lower:
                groups['Temperature'].append(header)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def read_csv_data():
        """Read all data from CSV file."""
        nonlocal last_row_count
        
        data = {}
        headers = []
        
        try:
            with open(csv_file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                
                rows = list(reader)
                if len(rows) == last_row_count:
                    return None  # No new data
                
                last_row_count = len(rows)
                
                # Initialize data structure
                for header in headers:
                    data[header] = []
                
                # Read all rows
                for row in rows:
                    for header in headers:
                        try:
                            value = float(row[header]) if row[header] else float('nan')
                            data[header].append(value)
                        except (ValueError, KeyError):
                            data[header].append(float('nan'))
                
                return data, headers
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None
    
    def update_plot():
        """Update the plot with latest data."""
        nonlocal fig, axes
        
        result = read_csv_data()
        if result is None:
            return
        
        data, headers = result
        
        # Group columns
        groups = group_columns(headers)
        print(groups)
        if not groups:
            print("Warning: No recognizable column groups found")
            return
        
        # Determine time column
        time_col = None
        if 'Time' in groups and groups['Time']:
            time_col = groups['Time'][0]
        elif 'time' in headers:
            time_col = 'time'
        else:
            print("Warning: No time column found")
            return
        
        # Get time data and scale
        times = data.get(time_col, [])
        if not times:
            return
        
        max_time = max(times) if times else 0
        if max_time >= 300 * 60:  # 300 minutes -> hours
            times_scaled = [t / 3600 for t in times]
            time_unit = "Hours"
        elif max_time >= 100:  # after ~100 seconds -> minutes
            times_scaled = [t / 60 for t in times]
            time_unit = "Minutes"
        else:
            times_scaled = times
            time_unit = "Seconds"
        
        xlabel = f"Time ({time_unit.lower()})"
        
        # Create or update figure
        if fig is None:
            num_groups = len([g for g in groups.keys() if g != 'Time'])
            if num_groups == 0:
                return
            
            # Arrange subplots (2 columns, as many rows as needed)
            rows = (num_groups + 1) // 2
            cols = 2 if num_groups > 1 else 1
            fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
            fig.suptitle(f'Live Data from {os.path.basename(csv_file_path)}', fontsize=14)
            plt.ion()
            plt.show(block=False)
            
            # Flatten axes if needed
            if num_groups == 1:
                axes = [axes]
            elif rows == 1:
                axes = list(axes)
            else:
                axes = axes.flatten()
        
        # Plot each group
        ax_idx = 0
        colors = ['b-', 'r-', 'g-', 'm-', 'c-', 'y-', 'k-']
        markers = ['o', 's', '^', 'd', 'v', 'x']
        
        for group_name, columns in groups.items():
            if group_name == 'Time':
                continue
            
            if ax_idx >= len(axes):
                break
            
            ax = axes[ax_idx]
            ax.clear()
            ax.set_title(group_name)
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.3)
            
            # Determine ylabel based on group
            if group_name == 'OD':
                ax.set_ylabel('Voltage (V)')
            elif group_name == 'Temperature':
                ax.set_ylabel('Temperature (°C)')
            
            # Plot each column in the group
            for col_idx, col in enumerate(columns):
                if col not in data:
                    continue
                values = data[col]
                if not values:
                    continue
                
                color = colors[col_idx % len(colors)]
                marker = markers[col_idx % len(markers)] if len(columns) > 1 else None
                style = f'{color[0]}{marker}-' if marker else color
                
                ax.plot(times_scaled, values, style, linewidth=2, 
                       label=col, markersize=4 if marker else None)
            
            if len(columns) > 1:
                ax.legend(fontsize=9)
            ax_idx += 1
        
        # Hide unused subplots
        for i in range(ax_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def update_loop():
        """Continuously update the plot."""
        while True:
            try:
                update_plot()
                time.sleep(update_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in update loop: {e}")
                time.sleep(update_interval)
    
    # Initial plot
    update_plot()
    
    # Start update thread
    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Plotting stopped by user")
        if fig:
            plt.close(fig)


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python plot_csv_data.py <csv_file_path> [update_interval]")
        print("Example: python plot_csv_data.py bioreactor_data/data.csv 5.0")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    update_interval = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    
    plot_csv_data(csv_file, update_interval)


if __name__ == "__main__":
    main()
