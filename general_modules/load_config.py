#!/usr/bin/env python3

import os

def load_config(config_path):
    """Load configuration config.txt"""
    config = {}

    print(f"Loading configuration from {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            original_line = line
            line = line.strip()

            if not line or line.startswith('%'):
                continue

            # Handle comments
            if '#' in line:
                line = line.split('#')[0].strip()

            if not line:
                continue

            # Handle both tab and space separation
            if '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split()

            if len(parts) >= 2:
                key = parts[0].strip().lower()
                value = ' '.join(parts[1:]).strip()

                if key == 'reserved':
                    continue

                config[key] = parse_value(value)

    print(f"Configuration loaded with {len(config)} parameters")

    return config

def parse_value(value_str):
    """Parse string value to appropriate type"""
    value_str = value_str.strip()

    # Handle comma-separated values (e.g., gpu_ids)
    if ',' in value_str:
        parts = [part.strip() for part in value_str.split(',')]
        try:
            # Try to parse as array of numbers
            return [int(part) if '.' not in part else float(part) for part in parts]
        except ValueError:
            # Return as array of strings (lowercase)
            return [part.lower() for part in parts]

    # Handle arrays (space-separated values)
    if ' ' in value_str:
        parts = value_str.split()
        if len(parts) > 1:
            try:
                # Try to parse as array of numbers
                return [int(part) if '.' not in part else float(part) for part in parts]
            except ValueError:
                # Return as array of strings (lowercase)
                return [part.lower() for part in parts]

    # Handle boolean values
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'

    # Handle numeric values
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        # Return string value as lowercase
        return value_str.lower()

def display_config(config):
    """Display loaded configuration parameters"""
    print("\n" + "="*60)
    print("CONFIGURATION PARAMETERS")
    print("="*60)

    # Display all parameters in alphabetical order
    for key, value in sorted(config.items()):
        print(f"  {key:<20}: {value} ({type(value).__name__})")

    print("="*60)
    print(f"Total parameters loaded: {len(config)}")
    print("="*60 + "\n")

