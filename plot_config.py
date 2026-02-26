"""
Configuration file for plot_csv_data.py remote data fetching.

Specifies SSH servers and file paths for fetching bioreactor data.
"""

# Default configuration
DEFAULT_FILENAME = "bioreactor_data.csv"
DEFAULT_REMOTE_PATH = "/home/david/Documents/GitHub/gluon/src/bioreactor_data/"

# SSH Server Configuration
# Each server entry should have:
#   - host: SSH hostname or IP address (e.g., "bioreactor00", "bioreactor00.local", or "192.168.1.100")
#   - user: SSH username (e.g., "david")
#   - remote_path: Path on remote server (default: DEFAULT_REMOTE_PATH)
#   - filename: Filename on remote server (default: DEFAULT_FILENAME)
#   - label: Optional label for this server (used in plots, defaults to host)
#
# Note: If hostname resolution fails, try:
#   - Using an IP address instead (e.g., "192.168.1.100")
#   - Using a fully qualified domain name (e.g., "bioreactor00.local" for mDNS/Bonjour)
#   - Adding an entry to /etc/hosts: "192.168.1.100 bioreactor00"

SSH_SERVERS = [
    {
        'host': 'bioreactor00',
        'user': 'david',
        'remote_path': DEFAULT_REMOTE_PATH,
        'filename': DEFAULT_FILENAME,
        'label': 'bioreactor00',  # Optional: custom label for plots
    },
    {
        'host': 'bioreactor01',
        'user': 'david',
        'remote_path': DEFAULT_REMOTE_PATH,
        'filename': DEFAULT_FILENAME,
        'label': 'bioreactor01',  # Optional: custom label for plots
    },
]

# SSH Configuration
SSH_TIMEOUT = 10  # Timeout in seconds for SSH connections
SSH_KEY_PATH = None  # Path to SSH private key (None = use default ~/.ssh/id_rsa)

# Local cache directory for downloaded files
CACHE_DIR = "/tmp/plot_csv_cache"  # Directory to cache remote files locally

# EKF subplot: 'doubling_time' (default) or 'growth_rate'
EKF_PLOT_MODE = 'doubling_time'

# Columns to exclude from plotting
# Use substrings that match column names (case-insensitive), e.g. 'ref' excludes 'OD_ref_V'
EXCLUDE_COLUMNS = ['ref']
