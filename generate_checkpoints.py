#!/usr/bin/env python3
"""Convenient CLI script to generate Town03 checkpoints using CARLA's map API.

Usage:
    python generate_checkpoints.py [--host localhost] [--port 2000] [--town Town03] [--output config/checkpoints_town03.json]
"""

import sys
from navigation.checkpoints import main

if __name__ == "__main__":
    if "--generate" not in sys.argv and "--verify" not in sys.argv:
        # Default to generation mode when run directly
        sys.argv.insert(1, "--generate")
    main()

