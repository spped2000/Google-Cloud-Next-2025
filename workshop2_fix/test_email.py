#!/usr/bin/env python3
"""
Standalone Email Agent Server
Run this script to start the Email Agent A2A server
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Get SMTP configuration from environment
    smtp_config = {
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "from_email": os.getenv("FROM_EMAIL"),
        "username": os.getenv("SMTP_USERNAME"),
        "password": os.getenv("SMTP_PASSWORD")
    }
    
    # Check required SMTP settings
    if not smtp_config["from_email"] or not smtp_config["username"] or not smtp_config["password"]:
        print("❌ Error: SMTP configuration incomplete!")
        print("Please set the following environment variables:")
        print("- FROM_EMAIL")
        print("- SMTP_USERNAME") 
        print("- SMTP_PASSWORD")
        print("Optional: SMTP_SERVER (default: smtp.gmail.com)")
        print("Optional: SMTP_PORT (default: 587)")
        sys.exit(1)
    
    # Import and run the server
    try:
        from multi_agent_system import run_email_agent_server
        run_email_agent_server(smtp_config)
    except ImportError as e:
        print(f"❌ Error importing multi_agent_system: {e}")
        print("Make sure multi_agent_system.py is in the same directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()