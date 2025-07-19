#!/usr/bin/env python3
"""
Quick OCR Test Script
Test the OCR functionality directly without A2A servers
"""

import os
import sys
import time
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

def test_ocr_directly():
    """Test OCR processing directly"""
    
    # Get API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment")
        return False
    
    # Test image path
    image_path = "th.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return False
    
    try:
        print("🔄 Testing OCR processing directly...")
        start_time = time.time()
        
        # Import the OCR function
        from multi_agent_system import process_ocr_image
        
        # Process OCR
        result = process_ocr_image(image_path, google_api_key)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        if result["status"] == "success":
            print("✅ OCR processing successful!")
            print(f"📊 Found {result['total_regions']} text regions")
            print("📝 Extracted text:")
            for i, text in enumerate(result['extracted_text'], 1):
                print(f"   {i}. {text}")
            return True
        else:
            print(f"❌ OCR processing failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Error during OCR test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_loading():
    """Test image loading and preprocessing"""
    image_path = "th.jpg"
    
    try:
        print("🔄 Testing image loading...")
        
        # Load image
        image = Image.open(image_path)
        print(f"📸 Image loaded: {image.width}x{image.height}, mode: {image.mode}")
        
        # Test resizing
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            original_size = (image.width, image.height)
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"📏 Resized from {original_size} to {image.width}x{image.height}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return False

def test_gemini_connection():
    """Test Gemini API connection"""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    try:
        print("🔄 Testing Gemini API connection...")
        
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=google_api_key)
        
        # Test simple text generation
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello, world! Just say 'Hi' back.",
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=10
            )
        )
        
        print(f"✅ Gemini API works! Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Quick OCR Test Suite")
    print("=" * 50)
    
    # Test 1: Gemini connection
    if not test_gemini_connection():
        print("❌ Gemini connection failed - check your API key")
        return False
    
    # Test 2: Image loading
    if not test_image_loading():
        print("❌ Image loading failed - check your image file")
        return False
    
    # Test 3: Direct OCR processing
    if not test_ocr_directly():
        print("❌ OCR processing failed")
        return False
    
    print("\n🎉 All tests passed!")
    print("✅ OCR system is working correctly")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)