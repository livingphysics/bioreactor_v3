import csv
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import List, Tuple, Optional, Union
from datetime import datetime

from . import components


class Bioreactor():
    """Class to manage all sensors and operations for the bioreactor"""
    
    def __init__(self, config=None) -> None:
        self.cfg = config  # Store config for access in utility functions
        """Initialize bioreactor framework without specific hardware components."""
        
        # Configuration
        self.cfg = config

        # Logging setup
        self.logger = logging.getLogger("Bioreactor")
        log_level = getattr(config, 'LOG_LEVEL', 'INFO') if config else 'INFO'
        self.logger.setLevel(getattr(logging, log_level))
        
        log_format = getattr(config, 'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s') if config else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Add file handler if LOG_FILE is specified
        if config and hasattr(config, 'LOG_FILE') and config.LOG_FILE:
            # Clear log file if CLEAR_LOG_ON_START is True
            clear_log = getattr(config, 'CLEAR_LOG_ON_START', False) if config else False
            if clear_log and os.path.exists(config.LOG_FILE):
                # Truncate the log file to clear it
                open(config.LOG_FILE, 'w').close()
            
            file_handler = logging.FileHandler(config.LOG_FILE)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Add terminal/console handler if LOG_TO_TERMINAL is True
        log_to_terminal = getattr(config, 'LOG_TO_TERMINAL', True) if config else True
        if log_to_terminal:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # If no handlers were added, add a default console handler
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.info("Initializing Bioreactor...")

        # Component initialization tracking
        self._initialized = {}
        
        # Initialize components based on config
        if config and hasattr(config, 'INIT_COMPONENTS'):
            self._init_components = config.INIT_COMPONENTS
            self._initialize_components(config)
        else:
            self._init_components = {}
            self.logger.warning("No component configuration found. No components will be initialized.")

        # Threading
        self._threads = []
        self._stop_event = threading.Event()

        # Set up CSV writer for sensor data
        # This can be customized based on your sensor configuration
        fieldnames = ['time']
        if config and hasattr(config, 'SENSOR_LABELS'):
            labels = config.SENSOR_LABELS
            # Include all configured labels
            fieldnames += [labels[k] for k in labels.keys()]
            
            # Ensure OD channels from OD_ADC_CHANNELS are present even if labels missing
            od_cfg = getattr(config, 'OD_ADC_CHANNELS', {}) or {}
            for ch in od_cfg.keys():
                label = (
                    labels.get(f"od_{ch}") or
                    labels.get(f"od_{ch.lower()}") or
                    labels.get(f"od_{ch.upper()}") or
                    f"OD_{ch}_V"
                )
                if label not in fieldnames:
                    fieldnames.append(label)
        else:
            # Default fieldnames if no config provided
            fieldnames = ['time']
        
        self.fieldnames = fieldnames
        
        # Get filename configuration
        base_filename = getattr(config, 'DATA_OUT_FILE', 'bioreactor_data.csv') if config else 'bioreactor_data.csv'
        use_timestamp = getattr(config, 'USE_TIMESTAMPED_FILENAME', True) if config else True
        
        # Ensure bioreactor_data directory exists
        data_dir = 'bioreactor_data'
        os.makedirs(data_dir, exist_ok=True)
        
        # Build filename with or without timestamp
        if use_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_file_path = os.path.join(data_dir, f"{timestamp}_{base_filename}")
        else:
            out_file_path = os.path.join(data_dir, base_filename)
        
        self.out_file = open(out_file_path, 'w', newline='')
        self.writer = csv.DictWriter(self.out_file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.logger.info(f"Data logging to: {out_file_path}")

        self.logger.info("Bioreactor initialization complete.")
        
        # Auto-flush on initialization if enabled (runs before pressurization)
        auto_flush = getattr(config, 'AUTO_FLUSH_ON_INIT', False) if config else False
        if auto_flush and self.is_component_initialized('relays'):
            try:
                from .utils import flush_tank
                flush_duration = getattr(config, 'AUTO_FLUSH_DURATION', 30.0) if config else 30.0
                self.logger.info(f"Starting automatic tank flush (duration: {flush_duration}s)...")
                flush_tank(self, duration_seconds=flush_duration)
                self.logger.info("Automatic tank flush completed.")
            except Exception as e:
                self.logger.error(f"Error during automatic tank flush: {e}")
        
        # Auto-pressurize on initialization if enabled
        auto_pressurize = getattr(config, 'AUTO_PRESSURIZE_ON_INIT', False) if config else False
        if auto_pressurize and self.is_component_initialized('relays'):
            try:
                from .utils import pressurize_and_inject_co2
                pressurize_duration = getattr(config, 'AUTO_PRESSURIZE_DURATION', 10.0) if config else 10.0
                pause = getattr(config, 'AUTO_PRESSURIZE_PAUSE', 30.0) if config else 30.0
                co2_duration = getattr(config, 'AUTO_PRESSURIZE_CO2_DURATION', 0.0) if config else 0.0
                self.logger.info(f"Starting automatic pressurization (duration: {pressurize_duration}s, pause: {pause}s, CO2: {co2_duration}s)...")
                pressurize_and_inject_co2(self, pressurize_duration=pressurize_duration, pause=pause, co2_duration=co2_duration)
                self.logger.info("Automatic pressurization completed.")
            except Exception as e:
                self.logger.error(f"Error during automatic pressurization: {e}")

    def _initialize_components(self, config) -> None:
        """Initialize components based on INIT_COMPONENTS configuration.
        
        Args:
            config: Configuration object with INIT_COMPONENTS dict
        """
        for component_name, should_init in self._init_components.items():
            if not should_init:
                self.logger.debug(f"Skipping {component_name} (disabled in config)")
                continue
            
            if component_name not in components.COMPONENT_REGISTRY:
                self.logger.warning(f"Component '{component_name}' not found in registry. Available: {list(components.COMPONENT_REGISTRY.keys())}")
                self._initialized[component_name] = False
                continue
            
            init_func = components.COMPONENT_REGISTRY[component_name]
            self.logger.info(f"Initializing {component_name}...")
            
            try:
                result = init_func(self, config)
                if result.get('initialized', False):
                    self._initialized[component_name] = True
                    self.logger.info(f"{component_name} initialized successfully")
                else:
                    self._initialized[component_name] = False
                    error = result.get('error', 'Unknown error')
                    self.logger.error(f"{component_name} initialization failed: {error}")
            except Exception as e:
                self._initialized[component_name] = False
                self.logger.error(f"{component_name} initialization exception: {e}")

    # Utility methods for component initialization tracking
    
    def register_component(self, component_name: str, initialized: bool = True) -> None:
        """Register a component as initialized or not.
        
        Args:
            component_name (str): Name of the component
            initialized (bool): Whether the component is initialized
        """
        self._initialized[component_name] = initialized
        if initialized:
            self.logger.info(f"{component_name} initialized.")
        else:
            self.logger.warning(f"{component_name} initialization failed.")
    
    def is_component_initialized(self, component_name: str) -> bool:
        """Check if a component is initialized.
        
        Args:
            component_name (str): Name of the component
            
        Returns:
            bool: True if initialized, False otherwise
        """
        return self._initialized.get(component_name, False)

    # Threaded scheduling

    def run(self, jobs):
        """
        Run jobs in separate threads.
        
        Args:
            jobs: list of (function, frequency, duration) tuples.
                - function: Function to call with self as first argument
                - frequency: Time in seconds between calls, or True for continuous
                - duration: How long to run in seconds, or True for indefinite
        
        Each function is called with self (or bioreactor_instance) as first argument.
        Non-blocking: returns immediately after starting threads.
        If duration is True, the job runs indefinitely until stop_all() is called.
        """
        self._stop_event.clear()

        self._threads = []
        def thread_worker(func, freq, dur):
            start = time.time()
            while not self._stop_event.is_set() and (dur is True or time.time() - start < dur):
                t0 = time.time()

                try:
                    global_elapsed = time.time() - start
                    func(self, elapsed=global_elapsed)
                except Exception as e:
                    self.logger.error(f"Exception in thread for {func.__name__}: {e}")

                if freq is True:
                    continue

                loop_elapsed = time.time() - t0
                sleep_time = max(0, freq - loop_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        for func, freq, dur in jobs:
            th = threading.Thread(target=thread_worker, args=(func, freq, dur))
            th.daemon = True
            th.start()
            self._threads.append(th)

        self.logger.info(f"Started {len(jobs)} job threads.")

    def stop_all(self):
        """Stop all running threads."""
        self._stop_event.set()
        self.logger.info("Stop event set for all threads.")

    def finish(self) -> None:
        """Clean up bioreactor resources."""
        self.logger.info("Cleaning up Bioreactor...")

        # Stop all threads
        self.stop_all()

        # Close CSV file
        if hasattr(self, 'out_file') and self.out_file:
            try:
                self.out_file.close()
                self.logger.info("Data file closed.")
            except Exception as e:
                self.logger.error(f"Error closing data file: {e}")

        # Stop Peltier PWM if active
        driver = getattr(self, 'peltier_driver', None)
        if driver:
            try:
                driver.stop()
            except Exception as e:
                self.logger.error(f"Failed to stop peltier driver: {e}")

        # Stop stirrer PWM if active
        stirrer_driver = getattr(self, 'stirrer_driver', None)
        if stirrer_driver:
            try:
                stirrer_driver.stop()
            except Exception as e:
                self.logger.error(f"Failed to stop stirrer driver: {e}")

        self.logger.info("Bioreactor cleanup complete.")

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        self.finish()
        return False

