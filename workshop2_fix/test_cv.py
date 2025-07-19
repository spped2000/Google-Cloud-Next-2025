#!/usr/bin/env python3
"""
Standalone CV Agent Server
Run this script to start the Computer Vision Agent A2A server
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Get API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set!")
        print("Please set it in your .env file or export it:")
        print("export GOOGLE_API_KEY='your_google_api_key_here'")
        sys.exit(1)
    
    # Import server creation function and uvicorn
    try:
        from multi_agent_system import create_cv_agent_server
        import uvicorn

        server, host, port = create_cv_agent_server(google_api_key)
        
        print(f"🚀 Starting Computer Vision Agent server on {host}:{port}")
        print(f"📍 Agent card available at: http://{host}:{port}/.well-known/agent.json")
        print(f"🔑 Using API key: {google_api_key[:10]}...{google_api_key[-5:]}")
        
        # Run the server
        uvicorn.run(server.build(), host=host, port=port)

    except ImportError as e:
        print(f"❌ Error importing from multi_agent_system: {e}")
        print("   Make sure multi_agent_system.py is in the same directory or accessible in PYTHONPATH.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()