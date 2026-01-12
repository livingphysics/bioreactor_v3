#!/usr/bin/env python3
"""
Standalone script to plot CSV data from bioreactor data files.

Reads CSV files from remote SSH servers (configured in plot_config.py) or local files
and plots data with automatic grouping of similar columns.
Groups columns by type (OD, Temperature) and updates the plot periodically.

Usage:
    python plot_csv_data.py [csv_file_path] [update_interval]
    
    If csv_file_path is not provided, will fetch from remote servers configured in plot_config.py.
    
Examples:
    python plot_csv_data.py bioreactor_data/20251210_134704_bioreactor_data.csv 5.0
    python plot_csv_data.py 5.0  # Use remote servers from config
"""

import csv
import os
import sys
import time
import threading
import tempfile
import shutil
import subprocess
import socket
import matplotlib.pyplot as plt
import numpy as np

# Try to import paramiko for SSH, fall back to subprocess if not available
try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
    print("Warning: paramiko not installed. Using subprocess for SSH (slower). Install with: pip install paramiko")

# Import config
try:
    import plot_config
except ImportError:
    print("Error: plot_config.py not found. Please create it with SSH server configuration.")
    sys.exit(1)


def fetch_remote_file(server_config, cache_dir):
    """
    Fetch a CSV file from a remote SSH server.
    
    Args:
        server_config: Dictionary with 'host', 'user', 'remote_path', 'filename', 'label'
        cache_dir: Local directory to cache the file
        
    Returns:
        Path to local cached file, or None if fetch failed
    """
    remote_file = os.path.join(server_config['remote_path'], server_config['filename']).replace('\\', '/')
    local_file = os.path.join(cache_dir, f"{server_config['label']}_{server_config['filename']}")
    
    try:
        if HAS_PARAMIKO:
            # Use paramiko for SSH
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            key_path = plot_config.SSH_KEY_PATH or os.path.expanduser("~/.ssh/id_rsa")
            key = None
            if os.path.exists(key_path):
                try:
                    key = paramiko.RSAKey.from_private_key_file(key_path)
                except:
                    pass
            
            ssh.connect(
                server_config['host'],
                username=server_config['user'],
                pkey=key,
                timeout=plot_config.SSH_TIMEOUT
            )
            
            sftp = ssh.open_sftp()
            try:
                sftp.get(remote_file, local_file)
            except FileNotFoundError:
                print(f"Warning: File not found on {server_config['host']}: {remote_file}")
                return None
            finally:
                sftp.close()
                ssh.close()
        else:
            # Fallback to subprocess with scp
            remote_path = f"{server_config['user']}@{server_config['host']}:{remote_file}"
            result = subprocess.run(
                ['scp', '-o', 'StrictHostKeyChecking=no', '-o', f'ConnectTimeout={plot_config.SSH_TIMEOUT}',
                 remote_path, local_file],
                capture_output=True,
                timeout=plot_config.SSH_TIMEOUT + 5
            )
            if result.returncode != 0:
                stderr = result.stderr.decode() if result.stderr else ""
                stdout = result.stdout.decode() if result.stdout else ""
                error_msg = stderr or stdout
                
                # Check for hostname resolution errors
                if "Name or service not known" in error_msg or "Could not resolve hostname" in error_msg:
                    host = server_config['host']
                    print(f"Error: Cannot resolve hostname '{host}'")
                    print(f"  Hint: Try using an IP address or fully qualified domain name (FQDN)")
                    print(f"  Example: Use 'bioreactor00.local' or '192.168.1.100' instead of 'bioreactor00'")
                else:
                    print(f"Warning: Failed to fetch from {server_config['host']}: {error_msg}")
                return None
        
        return local_file if os.path.exists(local_file) else None
        
    except paramiko.ssh_exception.SSHException as e:
        print(f"Error: SSH connection failed to {server_config['host']}: {e}")
        return None
    except socket.gaierror as e:
        host = server_config['host']
        print(f"Error: Cannot resolve hostname '{host}': {e}")
        print(f"  Hint: Try using an IP address or fully qualified domain name (FQDN)")
        print(f"  Example: Use 'bioreactor00.local' or '192.168.1.100' instead of 'bioreactor00'")
        return None
    except Exception as e:
        error_msg = str(e)
        if "Name or service not known" in error_msg or "Errno -2" in error_msg:
            host = server_config['host']
            print(f"Error: Cannot resolve hostname '{host}'")
            print(f"  Hint: The hostname cannot be resolved. Try:")
            print(f"    1. Use an IP address instead (e.g., '192.168.1.100')")
            print(f"    2. Use a fully qualified domain name (e.g., 'bioreactor00.local' or 'bioreactor00.example.com')")
            print(f"    3. Add an entry to /etc/hosts: '192.168.1.100 bioreactor00'")
        else:
            print(f"Error fetching from {server_config['host']}: {e}")
        return None


def fetch_all_remote_files(servers, cache_dir):
    """
    Fetch CSV files from all configured remote servers.
    
    Args:
        servers: List of server configuration dictionaries
        cache_dir: Local directory to cache files
        
    Returns:
        List of local file paths (None for failed fetches)
    """
    os.makedirs(cache_dir, exist_ok=True)
    local_files = []
    
    for server in servers:
        local_file = fetch_remote_file(server, cache_dir)
        local_files.append((server['label'], local_file))
    
    return local_files


def combine_csv_files(file_list):
    """
    Combine multiple CSV files into a single data structure.
    Adds a 'source' column to identify which server each row came from.
    
    Args:
        file_list: List of tuples (label, file_path) where file_path may be None
        
    Returns:
        Tuple of (combined_data dict, headers list)
    """
    all_data = {}
    all_headers = set()
    source_column = []
    
    # First pass: collect all headers
    for label, file_path in file_list:
        if file_path is None or not os.path.exists(file_path):
            continue
        try:
            with open(file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    all_headers.update(reader.fieldnames)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if 'source' not in all_headers:
        all_headers.add('source')
    
    all_headers = sorted(list(all_headers))
    
    # Initialize data structure
    for header in all_headers:
        all_data[header] = []
    
    # Second pass: read and combine data
    for label, file_path in file_list:
        if file_path is None or not os.path.exists(file_path):
            continue
        
        try:
            with open(file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                row_count = 0
                for row in reader:
                    row_count += 1
                    for header in all_headers:
                        if header == 'source':
                            all_data['source'].append(label)
                        else:
                            try:
                                value = float(row[header]) if header in row and row[header] else float('nan')
                                all_data[header].append(value)
                            except (ValueError, KeyError):
                                all_data[header].append(float('nan'))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return all_data, all_headers


def plot_csv_data(csv_file_path: str = None, update_interval: float = 5.0, use_remote: bool = False):
    """
    Read CSV file(s) and plot data with automatic grouping of similar columns.
    
    Can read from:
    - A single local CSV file (if csv_file_path is provided)
    - Multiple remote CSV files via SSH (if use_remote=True or csv_file_path is None)
    
    Groups columns by type:
    - OD and Eyespy voltage readings (columns containing 'OD', 'od', 'eyespy', or 'Eyespy') -> one subplot
    - Temperature (columns containing 'temp' or 'temperature') -> one subplot
    - Time -> x-axis for all
    
    Note: Only voltage columns are plotted (raw ADC values are excluded).
    
    Updates the plot periodically by re-reading the CSV file(s).
    
    Args:
        csv_file_path: Path to a local CSV file to read (optional if using remote)
        update_interval: Time in seconds between plot updates (default: 5.0)
        use_remote: If True, fetch from remote servers instead of using csv_file_path
    """
    # Determine if we're using remote files
    if csv_file_path is None or use_remote:
        use_remote = True
        servers = getattr(plot_config, 'SSH_SERVERS', [])
        if not servers:
            print("Error: No SSH servers configured in plot_config.py")
            return
        cache_dir = getattr(plot_config, 'CACHE_DIR', '/tmp/plot_csv_cache')
        print(f"Fetching data from {len(servers)} remote server(s)...")
    else:
        use_remote = False
        if csv_file_path is None or not os.path.exists(csv_file_path):
            print(f"Error: CSV file not found: {csv_file_path}")
            return
    
    # Global storage for plot data
    last_row_count = 0
    fig = None
    axes = None
    cache_dir = getattr(plot_config, 'CACHE_DIR', '/tmp/plot_csv_cache') if use_remote else None
    
    def group_columns(headers):
        """Group column headers by type."""
        groups = {
            'OD': [],
            'Temperature': [],
            'Time': []
        }
        
        for header in headers:
            # Skip 'source' column (used for identifying remote servers)
            if header.lower() == 'source':
                continue
            header_lower = header.lower()
            if header_lower == 'time':
                groups['Time'].append(header)
            elif 'od' in header_lower or 'eyespy' in header_lower:
                # Group both OD and eyespy voltage columns together
                # Only include voltage columns (not raw ADC values)
                if 'raw' not in header_lower:
                    groups['OD'].append(header)
            elif 'temp' in header_lower:
                groups['Temperature'].append(header)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def read_csv_data():
        """Read all data from CSV file(s)."""
        nonlocal last_row_count
        
        if use_remote:
            # Fetch from remote servers
            servers = getattr(plot_config, 'SSH_SERVERS', [])
            file_list = fetch_all_remote_files(servers, cache_dir)
            
            # Combine data from all files
            data, headers = combine_csv_files(file_list)
            
            # Check if we have new data
            total_rows = len(data.get('source', [])) if data else 0
            if total_rows == last_row_count:
                return None  # No new data
            
            last_row_count = total_rows
            return data, headers
        else:
            # Read from local file
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
            
            # Set plot title based on mode
            if use_remote:
                server_names = ', '.join([s['label'] for s in getattr(plot_config, 'SSH_SERVERS', [])])
                fig.suptitle(f'Live Data from Remote Servers ({server_names})', fontsize=14)
            else:
                fig.suptitle(f'Live Data from {os.path.basename(csv_file_path)}', fontsize=14)
            
            plt.ion()
            
            # Flatten axes if needed
            if num_groups == 1:
                axes = [axes]
            elif rows == 1:
                axes = list(axes)
            else:
                axes = axes.flatten()
            
            # Show the figure window
            plt.show(block=False)
            plt.pause(0.1)  # Give matplotlib time to display the window
        
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
                ax.set_ylabel('Voltage (V)')  # OD and Eyespy both use this
            elif group_name == 'Temperature':
                ax.set_ylabel('Temperature (°C)')
            
            # Plot each column in the group
            # If we have a 'source' column, plot each source separately
            if 'source' in data and use_remote:
                # Group by source and plot separately
                sources = set(data['source'])
                for source_idx, source in enumerate(sorted(sources)):
                    # Filter data for this source
                    source_indices = [i for i, s in enumerate(data['source']) if s == source]
                    source_times = [times_scaled[i] for i in source_indices]
                    
                    for col_idx, col in enumerate(columns):
                        if col not in data:
                            continue
                        source_values = [data[col][i] for i in source_indices]
                        if not source_values:
                            continue
                        
                        # Use different colors/markers for each source+column combination
                        style_idx = source_idx * len(columns) + col_idx
                        color = colors[style_idx % len(colors)]
                        marker = markers[style_idx % len(markers)] if len(sources) > 1 or len(columns) > 1 else None
                        style = f'{color[0]}{marker}-' if marker else color
                        
                        label = f"{source}: {col}" if len(sources) > 1 or len(columns) > 1 else col
                        ax.plot(source_times, source_values, style, linewidth=2, 
                               label=label, markersize=4 if marker else None)
            else:
                # Original behavior: plot all columns
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
            
            # Show legend if we have multiple series
            if (use_remote and 'source' in data and len(set(data['source'])) > 1) or len(columns) > 1:
                ax.legend(fontsize=9)
            ax_idx += 1
        
        # Hide unused subplots
        for i in range(ax_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Increased pause time to ensure display updates
    
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
    print("Reading CSV and creating plot...")
    update_plot()
    
    if fig is None:
        print("Warning: No data found to plot. Check that the CSV file has data.")
        return
    
    print("Plot window opened. Press Ctrl+C to stop.")
    
    # Start update thread
    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()
    
    # Keep main thread alive and process GUI events
    try:
        while True:
            plt.pause(0.1)  # Process GUI events
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nPlotting stopped by user")
        if fig:
            plt.close(fig)


def main():
    """Main entry point for command-line usage."""
    use_remote = False
    csv_file = None
    update_interval = 5.0
    
    # Parse arguments
    if len(sys.argv) < 2:
        # No arguments: use remote servers from config
        use_remote = True
    elif len(sys.argv) == 2:
        # One argument: could be file path or update interval
        try:
            update_interval = float(sys.argv[1])
            use_remote = True  # If it's a number, assume remote mode
        except ValueError:
            csv_file = sys.argv[1]  # If not a number, assume it's a file path
    elif len(sys.argv) == 3:
        # Two arguments: could be (file, interval) or (interval, interval)
        try:
            # Try to parse first as float
            float(sys.argv[1])
            # If successful, both are numbers - use remote
            use_remote = True
            update_interval = float(sys.argv[1])
        except ValueError:
            # First is file path, second is interval
            csv_file = sys.argv[1]
            update_interval = float(sys.argv[2])
    else:
        print("Usage: python plot_csv_data.py [csv_file_path] [update_interval]")
        print("  If csv_file_path is omitted, fetches from remote servers in plot_config.py")
        print("Examples:")
        print("  python plot_csv_data.py                                    # Use remote servers, 5s interval")
        print("  python plot_csv_data.py 10.0                               # Use remote servers, 10s interval")
        print("  python plot_csv_data.py data.csv                           # Use local file, 5s interval")
        print("  python plot_csv_data.py data.csv 10.0                      # Use local file, 10s interval")
        sys.exit(1)
    
    plot_csv_data(csv_file, update_interval, use_remote)


if __name__ == "__main__":
    main()
