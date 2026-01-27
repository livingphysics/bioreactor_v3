#!/usr/bin/env python3
"""
Standalone script to plot CSV data from bioreactor data files.

Supports two modes:
1. LOCAL MODE: Plot data from a single local CSV file
2. REMOTE MODE: Fetch and plot data from multiple remote SSH servers

Reads CSV files and plots data with automatic grouping of similar columns.
Groups columns by type (OD, Temperature) and updates the plot periodically.
Each bioreactor appears in its own row when using remote mode.

Usage:
    python plot_csv_data.py [options] [csv_file_path] [update_interval]
    
Options:
    --remote, -r    Force remote mode (fetch from SSH servers)
    --local, -l    Force local mode (read from local file)
    
Examples:
    # Remote mode (default when no file specified):
    python plot_csv_data.py                    # Remote, 5s interval
    python plot_csv_data.py --remote 10.0     # Remote, 10s interval
    
    # Local mode:
    python plot_csv_data.py data.csv          # Local file, 5s interval
    python plot_csv_data.py data.csv 10.0    # Local file, 10s interval
    python plot_csv_data.py --local data.csv  # Explicitly local mode
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
import queue
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend explicitly
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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


def plot_csv_data(csv_file_path: str = None, update_interval: float = 5.0, use_remote: bool = False, debug: bool = False):
    """
    Read CSV file(s) and plot data with automatic grouping of similar columns.
    
    Supports two modes:
    
    1. LOCAL MODE: Read from a single local CSV file
       - Provide csv_file_path pointing to a local CSV file
       - Set use_remote=False (or omit csv_file_path and use_remote)
       - Example: plot_csv_data('data.csv', 5.0, False)
    
    2. REMOTE MODE: Fetch from multiple remote SSH servers
       - Set use_remote=True (or omit csv_file_path)
       - Configure servers in plot_config.py
       - Fetches CSV files from all configured servers and combines them
       - Each bioreactor appears in its own row
       - Example: plot_csv_data(None, 5.0, True)
    
    Groups columns by type:
    - OD and Eyespy voltage readings (columns containing 'OD', 'od', 'eyespy', or 'Eyespy') -> one subplot
    - Temperature (columns containing 'temp' or 'temperature') -> one subplot
    - Gases (CO2 and O2 columns containing 'co2', 'CO2', 'o2', or 'O2') -> one subplot with dual y-axes
    - Time -> x-axis for all
    
    Note: Only voltage columns are plotted (raw ADC values are excluded).
    
    Updates the plot periodically by re-reading the CSV file(s).
    
    Args:
        csv_file_path: Path to a local CSV file to read (required for local mode, ignored for remote mode)
        update_interval: Time in seconds between plot updates (default: 5.0)
        use_remote: If True, fetch from remote servers configured in plot_config.py
                   If False and csv_file_path provided, read from local file
                   If False and csv_file_path is None, defaults to remote mode
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
    twin_axes = {}  # Store twin axes by (source_idx, group_idx) to reuse them
    cache_dir = getattr(plot_config, 'CACHE_DIR', '/tmp/plot_csv_cache') if use_remote else None
    update_queue = queue.Queue()  # Queue for thread-safe updates
    update_flag = threading.Event()  # Flag to signal updates from background thread
    
    def group_columns(headers):
        """Group column headers by type."""
        groups = {
            'OD': [],
            'Temperature': [],
            'Gases': [],
            'Time': []
        }
        
        for header in headers:
            # Skip 'source' column (used for identifying remote servers)
            if header.lower() == 'source':
                continue
            header_lower = header.lower()
            if header_lower == 'time' or header_lower == 'elapsed_time':
                groups['Time'].append(header)
            elif 'od' in header_lower or 'eyespy' in header_lower:
                # Group both OD and eyespy voltage columns together
                # Only include voltage columns (not raw ADC values)
                if 'raw' not in header_lower:
                    groups['OD'].append(header)
            elif 'temp' in header_lower:
                groups['Temperature'].append(header)
            elif 'co2' in header_lower:
                # Group CO2 columns (e.g., CO2_ppm)
                groups['Gases'].append(header)
            elif 'o2' in header_lower and 'co2' not in header_lower:
                # Group O2 columns (e.g., O2_percent) - add to Gases group for dual-axis plotting
                groups['Gases'].append(header)
        
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
    
    def update_plot(data=None, headers=None):
        """Update the plot with latest data. Must be called from main thread."""
        nonlocal fig, axes, twin_axes
        
        # If data/headers not provided, read them (for initial call)
        if data is None or headers is None:
            result = read_csv_data()
            if result is None:
                return
            data, headers = result
        
        # Group columns (this automatically removes empty groups)
        groups = group_columns(headers)
        if not groups:
            print("Warning: No recognizable column groups found")
            return
        
        # Debug: show which groups were found (after filtering empty ones)
        if debug:
            print(f"DEBUG: Groups found after filtering: {list(groups.keys())}")
        
        # Check if we have a Time group (required)
        if 'Time' not in groups or not groups['Time']:
            print("Warning: No time column found")
            return
        
        # Determine time column (prefer 'elapsed_time' for plotting as it contains elapsed seconds, fall back to time)
        time_col = None
        if 'Time' in groups and groups['Time']:
            # Prefer 'elapsed_time' if available (contains elapsed seconds), otherwise use first time column
            if 'elapsed_time' in groups['Time']:
                time_col = 'elapsed_time'
            else:
                time_col = groups['Time'][0]
        elif 'elapsed_time' in headers:
            time_col = 'elapsed_time'
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
        
        # Determine sources (bioreactors) - if using remote, group by source; otherwise single source
        if 'source' in data and use_remote:
            sources = sorted(set(data['source']))
        else:
            sources = ['Local']  # Single source for local files
        
        # Get data type groups (excluding Time) - only include non-empty groups
        data_groups = {k: v for k, v in groups.items() if k != 'Time' and v}  # Ensure groups are non-empty
        num_groups = len(data_groups)
        
        if num_groups == 0:
            print("Warning: No data groups found (only Time column present)")
            return
        
        # Calculate number of sources (needed for both figure creation and plotting)
        num_sources = len(sources)
        
        # Check if figure needs to be recreated (if group structure changed)
        current_group_names = sorted(data_groups.keys())
        if fig is not None:
            # Check if we need to recreate the figure due to structure change
            # Compare both the number of groups and the actual group names
            expected_cols = len(current_group_names)
            last_group_names = getattr(fig, '_last_group_names', None)
            if (hasattr(fig, '_last_num_groups') and fig._last_num_groups != expected_cols) or \
               (last_group_names is not None and last_group_names != current_group_names):
                # Structure changed, close and recreate
                plt.close(fig)
                fig = None
                axes = None
        
        # Create or update figure
        # Layout: Each bioreactor (source) gets a row, each row has subplots for each data type
        if fig is None:
            num_cols = num_groups  # One column per data type
            num_rows = num_sources  # One row per bioreactor
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 4 * num_rows))
            
            # Set plot title based on mode
            if use_remote:
                server_names = ', '.join([s['label'] for s in getattr(plot_config, 'SSH_SERVERS', [])])
                fig.suptitle(f'Live Data from Remote Servers ({server_names})', fontsize=14)
            else:
                fig.suptitle(f'Live Data from {os.path.basename(csv_file_path)}', fontsize=14)
            
            plt.ion()
            
            # Normalize axes to always be a 2D list for consistent access
            # matplotlib's subplots returns different structures depending on dimensions
            # Note: np is already imported at module level
            
            if num_rows == 1 and num_cols == 1:
                # Single subplot: axes is a single Axes object
                axes = [[axes]]
            elif num_rows == 1:
                # One row, multiple columns: axes is a 1D array, convert to list
                if isinstance(axes, np.ndarray):
                    axes = [axes[i] for i in range(num_cols)]  # Convert to list of axes
                elif not isinstance(axes, list):
                    axes = [axes]
                axes = [axes]  # Make it 2D: [row0] where row0 is a list of axes
            elif num_cols == 1:
                # Multiple rows, one column: axes is a 1D array, convert to list
                if isinstance(axes, np.ndarray):
                    axes = [[axes[i]] for i in range(num_rows)]  # Convert to 2D list
                else:
                    axes = [[ax] for ax in axes]  # Make it 2D: [[row0], [row1], ...]
            else:
                # Multiple rows and columns: axes is a 2D array, convert to 2D list
                if isinstance(axes, np.ndarray):
                    axes = [[axes[i, j] for j in range(num_cols)] for i in range(num_rows)]
                elif not isinstance(axes[0], list):
                    # If it's not already a 2D list, convert it
                    axes = [[axes[i][j] for j in range(num_cols)] for i in range(num_rows)]
            
            # Show the figure window
            plt.show(block=False)
            plt.pause(0.1)  # Give matplotlib time to display the window
            
            # Store the number of groups and group names for change detection
            fig._last_num_groups = num_groups
            fig._last_group_names = current_group_names
        
        # Plot each bioreactor (source) in its own row
        colors = ['b-', 'r-', 'g-', 'm-', 'c-', 'y-', 'k-']
        markers = ['o', 's', '^', 'd', 'v', 'x']
        
        # Get sorted list of data groups for consistent column ordering
        group_names = sorted([g for g in data_groups.keys()])
        
        for source_idx, source in enumerate(sources):
            # Filter data for this source
            if 'source' in data and use_remote:
                source_indices = [i for i, s in enumerate(data['source']) if s == source]
                source_times = [times_scaled[i] for i in source_indices]
            else:
                source_indices = list(range(len(times_scaled)))
                source_times = times_scaled
            
            # Plot each data type group in a column for this source row
            for group_idx, group_name in enumerate(group_names):
                columns = data_groups[group_name]
                
                # Get the axis for this source row and group column
                # axes is now guaranteed to be 2D: axes[row][col]
                ax = axes[source_idx][group_idx]
                
                # Clear any existing twin axis for this subplot (but keep it for potential reuse)
                twin_key = (source_idx, group_idx)
                if twin_key in twin_axes:
                    old_ax2 = twin_axes[twin_key]
                    if old_ax2 in ax.figure.axes:
                        old_ax2.clear()  # Clear the twin axis but don't remove it yet
                
                ax.clear()
                
                # Set title: source name and data type
                if num_sources > 1:
                    ax.set_title(f'{source} - {group_name}')
                else:
                    ax.set_title(group_name)
                
                ax.set_xlabel(xlabel)
                ax.grid(True, alpha=0.3)
                
                # Separate CO2 and O2 columns for dual-axis plotting
                co2_columns = []
                o2_columns = []
                for col in columns:
                    col_lower = col.lower().strip()
                    # CO2 column: contains 'co2' and does NOT contain standalone 'o2' (without 'co2' before it)
                    # This matches: CO2_ppm, co2_ppm, CO2_ppm_x10, etc.
                    if 'co2' in col_lower:
                        # Make sure it's not an O2 column that happens to contain 'co2' as part of something else
                        # If it has 'o2' but 'co2' comes before 'o2', it's a CO2 column
                        # If it has 'o2' and 'co2' doesn't come before it, skip (shouldn't happen with our naming)
                        co2_columns.append(col)
                    # O2 column: contains 'o2' but NOT 'co2'
                    elif 'o2' in col_lower and 'co2' not in col_lower:
                        o2_columns.append(col)
                
                # Debug: print what columns we found
                if debug and group_name == 'Gases':
                    print(f"DEBUG: All columns in Gases group: {columns}")
                    print(f"DEBUG: CO2 columns found: {co2_columns}")
                    print(f"DEBUG: O2 columns found: {o2_columns}")
                
                # Determine ylabel and create secondary axis if needed
                ax2 = None
                if group_name == 'OD':
                    ax.set_ylabel('Voltage (V)')  # OD and Eyespy both use this
                elif group_name == 'Temperature':
                    ax.set_ylabel('Temperature (°C)')
                elif group_name == 'Gases':
                    # Check if we have valid O2 data before deciding on axis setup
                    has_valid_o2_data = False
                    if o2_columns:
                        for col in o2_columns:
                            if col in data:
                                source_values = [data[col][i] for i in source_indices]
                                valid_indices = [i for i, v in enumerate(source_values) if not np.isnan(v)]
                                if valid_indices:
                                    has_valid_o2_data = True
                                    break
                    
                    # Check if we have valid CO2 data
                    has_valid_co2_data = False
                    if co2_columns:
                        for col in co2_columns:
                            if col in data:
                                source_values = [data[col][i] for i in source_indices]
                                valid_indices = [i for i, v in enumerate(source_values) if not np.isnan(v)]
                                if valid_indices:
                                    has_valid_co2_data = True
                                    break
                    
                    # Handle CO2/O2 dual-axis plotting - only create dual axes if both have valid data
                    ax2 = None
                    twin_key = (source_idx, group_idx)
                    if has_valid_co2_data and has_valid_o2_data:
                        # Both CO2 and O2 have valid data: use dual axes
                        ax.set_ylabel('CO2 (ppm)', color='b')
                        ax.tick_params(axis='y', labelcolor='b')
                        # Create or reuse twin axis
                        if twin_key not in twin_axes:
                            ax2 = ax.twinx()
                            twin_axes[twin_key] = ax2
                        else:
                            ax2 = twin_axes[twin_key]
                        ax2.set_ylabel('O2 (%)', color='r')
                        ax2.tick_params(axis='y', labelcolor='r')
                        # Format O2 axis to show 2 decimal places, not scientific notation
                        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
                        # Set O2 axis range to 5% to 30% (will be reapplied after plotting)
                    elif has_valid_co2_data:
                        # Only CO2 has valid data: use primary axis
                        ax.set_ylabel('CO2 (ppm)')
                        # Remove twin axis if it exists (no longer needed)
                        if twin_key in twin_axes:
                            old_ax2 = twin_axes[twin_key]
                            if old_ax2 in ax.figure.axes:
                                old_ax2.remove()
                            del twin_axes[twin_key]
                    elif has_valid_o2_data:
                        # Only O2 has valid data: use primary axis
                        ax.set_ylabel('O2 (%)')
                        # Format O2 axis to show 2 decimal places, not scientific notation
                        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
                        # Set O2 axis range to 5% to 30%
                        ax.set_ylim(5.0, 30.0)
                        # Remove twin axis if it exists (no longer needed)
                        if twin_key in twin_axes:
                            old_ax2 = twin_axes[twin_key]
                            if old_ax2 in ax.figure.axes:
                                old_ax2.remove()
                            del twin_axes[twin_key]
                    else:
                        # Neither has valid data: default to CO2 label
                        ax.set_ylabel('Gases')
                        # Remove twin axis if it exists (no longer needed)
                        if twin_key in twin_axes:
                            old_ax2 = twin_axes[twin_key]
                            if old_ax2 in ax.figure.axes:
                                old_ax2.remove()
                            del twin_axes[twin_key]
                
                # Plot CO2 columns on primary axis
                for col_idx, col in enumerate(co2_columns):
                    if col not in data:
                        if debug:
                            print(f"DEBUG: CO2 column '{col}' not found in data dictionary. Available keys: {list(data.keys())[:10]}")
                        continue
                    source_values = [data[col][i] for i in source_indices]
                    # Filter out NaN values for plotting
                    valid_indices = [i for i, v in enumerate(source_values) if not np.isnan(v)]
                    if not valid_indices:
                        if debug:
                            print(f"DEBUG: CO2 column '{col}' has no valid (non-NaN) values")
                        continue
                    
                    # Use only valid data points
                    valid_times = [source_times[i] for i in valid_indices]
                    valid_values = [source_values[i] for i in valid_indices]
                    
                    # For CO2_ppm_x10, divide by 10 to get actual ppm for display
                    if 'ppm_x10' in col.lower() or 'ppm_x' in col.lower():
                        valid_values = [v / 10.0 for v in valid_values]
                    
                    color = colors[col_idx % len(colors)]
                    marker = markers[col_idx % len(markers)] if len(co2_columns) > 1 else None
                    style = f'{color[0]}{marker}-' if marker else color
                    
                    label = col.replace('_ppm_x10', ' (ppm)').replace('_x10', '')
                    ax.plot(valid_times, valid_values, style, linewidth=2, 
                           label=label, markersize=4 if marker else None)
                
                # Plot O2 columns (on secondary axis if both exist, otherwise primary)
                if o2_columns:
                    target_ax = ax2 if ax2 is not None else ax
                    for col_idx, col in enumerate(o2_columns):
                        if col not in data:
                            if debug:
                                print(f"DEBUG: O2 column '{col}' not found in data dictionary. Available keys: {list(data.keys())[:10]}")
                            continue
                        source_values = [data[col][i] for i in source_indices]
                        # Filter out NaN values for plotting
                        valid_indices = [i for i, v in enumerate(source_values) if not np.isnan(v)]
                        if not valid_indices:
                            if debug:
                                print(f"DEBUG: O2 column '{col}' has no valid (non-NaN) values")
                            continue
                        
                        # Use only valid data points
                        valid_times = [source_times[i] for i in valid_indices]
                        valid_values = [source_values[i] for i in valid_indices]
                        
                        # Use red colors for O2
                        o2_colors = ['r', 'darkred', 'crimson', 'salmon']
                        color = o2_colors[col_idx % len(o2_colors)]
                        marker = markers[col_idx % len(markers)] if len(o2_columns) > 1 else None
                        style = f'{color}{marker}-' if marker else f'{color}-'
                        
                        label = col.replace('_percent', ' (%)').replace('_%', ' (%)')
                        target_ax.plot(valid_times, valid_values, style, linewidth=2, 
                               label=label, markersize=4 if marker else None)
                
                # Set O2 axis range and formatter after plotting (if O2 was plotted)
                if o2_columns:
                    # Determine which axis was used for O2
                    o2_axis = ax2 if ax2 is not None and has_valid_o2_data and has_valid_co2_data else (ax if has_valid_o2_data else None)
                    if o2_axis is not None:
                        # Set fixed range for O2 axis
                        o2_axis.set_ylim(5.0, 30.0)
                        # Ensure formatter is applied (reapply after plotting in case it was reset)
                        o2_axis.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
                
                # Show legend if we have multiple columns (combine CO2 and O2 legends)
                all_columns = co2_columns + o2_columns
                if len(all_columns) > 1:
                    # Combine legends from both axes if dual-axis, otherwise use single axis
                    if ax2 is not None:
                        lines1, labels1 = ax.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')
                    else:
                        ax.legend(fontsize=9)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Increased pause time to ensure display updates
    
    def update_loop():
        """Continuously read data and signal main thread to update plot."""
        while True:
            try:
                # Read data in background thread (this is safe)
                result = read_csv_data()
                if result is not None:
                    # Put data in queue for main thread to process
                    update_queue.put(result)
                    update_flag.set()  # Signal main thread
                time.sleep(update_interval)
            except KeyboardInterrupt:
                update_queue.put(None)  # Signal shutdown
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
    
    # Start update thread (reads data in background)
    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()
    
    # Main thread: process GUI events and handle plot updates
    try:
        while True:
            # Check for updates from background thread
            if update_flag.is_set():
                try:
                    # Get data from queue (non-blocking)
                    result = update_queue.get_nowait()
                    if result is None:  # Shutdown signal
                        break
                    data, headers = result
                    # Update plot on main thread (this is safe)
                    update_plot(data, headers)
                    update_flag.clear()
                except queue.Empty:
                    pass
            
            # Process GUI events (must be on main thread)
            plt.pause(0.1)
            time.sleep(0.1)  # Small sleep to prevent busy loop
    except KeyboardInterrupt:
        print("\nPlotting stopped by user")
    finally:
        if fig:
            plt.close(fig)


def main():
    """Main entry point for command-line usage."""
    use_remote = False
    csv_file = None
    update_interval = 5.0
    debug = False
    
    # Parse arguments
    # Check for explicit flags first
    if '--remote' in sys.argv or '-r' in sys.argv:
        use_remote = True
        # Remove flag from args for further processing
        sys.argv = [a for a in sys.argv if a not in ['--remote', '-r']]
    
    if '--local' in sys.argv or '-l' in sys.argv:
        use_remote = False
        # Remove flag from args for further processing
        sys.argv = [a for a in sys.argv if a not in ['--local', '-l']]
    
    if '--debug' in sys.argv or '-d' in sys.argv:
        debug = True
        # Remove flag from args for further processing
        sys.argv = [a for a in sys.argv if a not in ['--debug', '-d']]
    
    # Parse remaining arguments
    if len(sys.argv) < 2:
        # No arguments: use remote servers from config
        use_remote = True
    elif len(sys.argv) == 2:
        # One argument: could be file path or update interval
        try:
            update_interval = float(sys.argv[1])
            # If it's a number and no explicit flag, assume remote mode
            if not ('--local' in sys.argv or '-l' in sys.argv):
                use_remote = True
        except ValueError:
            csv_file = sys.argv[1]  # If not a number, assume it's a file path
            use_remote = False
    elif len(sys.argv) == 3:
        # Two arguments: could be (file, interval) or (interval, interval)
        try:
            # Try to parse first as float
            float(sys.argv[1])
            # If successful, both are numbers - use remote unless --local specified
            if not ('--local' in sys.argv or '-l' in sys.argv):
                use_remote = True
            update_interval = float(sys.argv[1])
        except ValueError:
            # First is file path, second is interval
            csv_file = sys.argv[1]
            update_interval = float(sys.argv[2])
            use_remote = False
    else:
        print("Usage: python plot_csv_data.py [options] [csv_file_path] [update_interval]")
        print("\nOptions:")
        print("  --remote, -r    Force remote mode (fetch from SSH servers)")
        print("  --local, -l     Force local mode (read from local file)")
        print("  --debug, -d     Enable debug output")
        print("\nModes:")
        print("  Remote mode: Fetches CSV files from SSH servers configured in plot_config.py")
        print("  Local mode:  Reads from a single local CSV file")
        print("\nExamples:")
        print("  # Remote mode (default when no file specified):")
        print("  python plot_csv_data.py                                    # Remote, 5s interval")
        print("  python plot_csv_data.py --remote 10.0                     # Remote, 10s interval")
        print("  python plot_csv_data.py -r                                 # Remote, 5s interval")
        print("\n  # Local mode:")
        print("  python plot_csv_data.py data.csv                           # Local file, 5s interval")
        print("  python plot_csv_data.py --local data.csv                  # Local file, 5s interval")
        print("  python plot_csv_data.py data.csv 10.0                     # Local file, 10s interval")
        print("  python plot_csv_data.py -l data.csv 10.0                  # Local file, 10s interval")
        sys.exit(1)
    
    plot_csv_data(csv_file, update_interval, use_remote, debug)


if __name__ == "__main__":
    main()
