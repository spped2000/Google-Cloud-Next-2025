import asyncio
import json
import requests
import time
import os
from typing import Dict, Any, List, Optional 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCREmailWorkflowTester:
    """Complete tester for OCR to Email workflow"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cv_agent_url = "http://localhost:5001"
        self.email_agent_url = "http://localhost:5002"
        
    def test_agent_availability(self) -> bool:
        """Test if both agents are running and available"""
        print("🔍 Step 1: Testing Agent Availability")
        
        # Test CV Agent
        try:
            response = requests.get(f"{self.cv_agent_url}/.well-known/agent.json", timeout=5)
            if response.status_code == 200:
                cv_card = response.json()
                print(f"✅ CV Agent available: {cv_card['name']}")
                print(f"   Skills: {[skill['name'] for skill in cv_card['skills']]}")
            else:
                print(f"❌ CV Agent not available (Status: {response.status_code})")
                return False
        except Exception as e:
            print(f"❌ CV Agent connection failed: {e}")
            return False
        
        # Test Email Agent
        try:
            response = requests.get(f"{self.email_agent_url}/.well-known/agent.json", timeout=5)
            if response.status_code == 200:
                email_card = response.json()
                print(f"✅ Email Agent available: {email_card['name']}")
                print(f"   Skills: {[skill['name'] for skill in email_card['skills']]}")
            else:
                print(f"❌ Email Agent not available (Status: {response.status_code})")
                return False
        except Exception as e:
            print(f"❌ Email Agent connection failed: {e}")
            return False
        
        print("✅ All agents are available!\n")
        return True
    
    def test_ocr_processing(self, image_path: str) -> Dict[str, Any]:
        """Test OCR processing on CV agent"""
        print("🔍 Step 2: Testing OCR Processing")
        
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return {"status": "error", "error": "Image file not found"}
        
        print(f"📷 Processing image: {image_path}")
        
        try:
            ocr_request = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": f"image_path: {image_path}"
                            }
                        ],
                        "messageId": f"msg_{int(time.time() * 1000)}"
                    }
                },
                "id": 1
            }
            
            print("📤 Sending OCR request to CV Agent...")
            response = requests.post(
                f"{self.cv_agent_url}/",
                json=ocr_request,
                headers={"Content-Type": "application/json"},
                timeout=120  # Increased timeout to 2 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ OCR processing completed!")
                
                ocr_results = self._parse_ocr_response(result)
                
                if ocr_results:
                    print(f"📊 Found {len(ocr_results.get('extracted_text', []))} text regions")
                    print("📝 Extracted text:")
                    for i, text in enumerate(ocr_results.get('extracted_text', []), 1):
                        print(f"   {i}. {text}")
                    
                    return {
                        "status": "success",
                        "ocr_results": ocr_results, 
                        "raw_response": result
                    }
                else:
                    print("❌ Failed to parse OCR results")
                    return {"status": "error", "error": "Failed to parse OCR results", "raw_response": result}
            else:
                print(f"❌ OCR request failed (Status: {response.status_code})")
                print(f"Response: {response.text}")
                return {"status": "error", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ OCR processing failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_email_sending(self, ocr_results: Dict[str, Any], recipients: List[str], image_path: str) -> Dict[str, Any]:
        """Test email sending with OCR results"""
        print("\n🔍 Step 3: Testing Email Sending")
        
        email_request_data = {
            "recipients": recipients,
            "subject": f"OCR Results - {len(ocr_results.get('extracted_text', []))} text regions detected",
            "ocr_results": ocr_results,
            "attachments": [],
            "image_path": image_path
        }
        
        email_request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": json.dumps(email_request_data)
                        }
                    ],
                    "messageId": f"msg_{int(time.time() * 1000)}"
                }
            },
            "id": 1
        }
        
        print(f"📧 Sending email to: {', '.join(recipients)}")
        
        try:
            response = requests.post(
                f"{self.email_agent_url}/",
                json=email_request,
                headers={"Content-Type": "application/json"},
                timeout=60  # Increased timeout to 1 minute
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Email sent successfully!")
                
                # Parse email response
                email_result = self._parse_email_response(result)
                return {
                    "status": "success",
                    "email_result": email_result,
                    "raw_response": result
                }
            else:
                print(f"❌ Email sending failed (Status: {response.status_code})")
                print(f"Response: {response.text}")
                return {"status": "error", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Email sending failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_full_workflow_test(self, image_path: str, recipients: List[str]) -> Dict[str, Any]:
        """Run the complete OCR to Email workflow test"""
        print("🚀 Starting Full OCR to Email Workflow Test")
        print("=" * 50)
        
        if not self.test_agent_availability():
            return {"status": "error", "error": "Agents not available"}
        
        ocr_result = self.test_ocr_processing(image_path)
        if ocr_result["status"] != "success":
            return ocr_result
        
        email_result = self.test_email_sending(ocr_result["ocr_results"], recipients, image_path)
        if email_result["status"] != "success":
            return email_result
        
        print("\n🎉 Full workflow test completed successfully!")
        print("=" * 50)
        
        return {
            "status": "success",
            "ocr_result": ocr_result,
            "email_result": email_result
        }
    
    def _parse_ocr_response(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse the JSON response from the A2A agent.
        It now expects the agent to return a JSON string with the full results.
        """
        try:
            # The agent's response is a JSON string inside result -> parts -> text
            if "result" in response and "parts" in response["result"]:
                for part in response["result"]["parts"]:
                    if part.get("kind") == "text" and "text" in part:
                        # The text part IS the JSON result string. We load it.
                        ocr_results = json.loads(part["text"])
                        
                        if ocr_results.get("status") == "success":
                            return ocr_results
                        else:
                            logger.error(f"CV Agent returned an error status: {ocr_results.get('error')}")
                            return None
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from CV Agent response: {response}")
            return None
        except Exception as e:
            logger.error(f"An exception occurred while parsing OCR response: {e}")
        
        logger.warning(f"Could not find or parse valid OCR data in response: {json.dumps(response)}")
        return None

    def _parse_email_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse email response to extract results"""
        try:
            if "result" in response:
                result = response["result"]
                
                if "artifacts" in result:
                    for artifact in result["artifacts"]:
                        if "parts" in artifact:
                            for part in artifact["parts"]:
                                if "text" in part:
                                    text = part["text"]
                                    if "Email Sent Successfully" in text:
                                        return {
                                            "status": "success",
                                            "message": "Email sent successfully",
                                            "response_text": text
                                        }
                
                if "history" in result:
                    for message in result["history"]:
                        if message.get("role") == "agent" and "parts" in message:
                            for part in message["parts"]:
                                if "text" in part:
                                    text = part["text"]
                                    if "Email Sent Successfully" in text:
                                        return {
                                            "status": "success",
                                            "message": "Email sent successfully",
                                            "response_text": text
                                        }
            
            return {"status": "unknown", "response": response}
        except Exception as e:
            logger.error(f"Error parsing email response: {e}")
            return {"status": "error", "error": str(e)}

def create_test_image():
    """Create a test image with Thai text (if PIL available)"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        
        draw.text((20, 20), "Hello World", fill='black', font=font)
        draw.text((20, 60), "This is a test image", fill='black', font=font)
        
        
        draw.text((20, 100), "สวัสดี", fill='black', font=font)
        draw.text((20, 140), "ข้อความทดสอบ", fill='black', font=font)
        
        
        test_image_path = "test_image.png"
        img.save(test_image_path)
        print(f"✅ Created test image: {test_image_path}")
        return test_image_path
    
    except ImportError:
        print("❌ PIL not available. Please provide an existing image file.")
        return None

def wait_for_servers():
    """Wait for servers to be ready"""
    print("⏳ Waiting for servers to be ready...")
    cv_ready = False
    email_ready = False
    
    for attempt in range(30):  
        try:
            if not cv_ready:
                response = requests.get("http://localhost:5001/.well-known/agent.json", timeout=2)
                if response.status_code == 200:
                    cv_ready = True
                    print("✅ CV Agent server ready")
            
            # Check Email agent
            if not email_ready:
                response = requests.get("http://localhost:5002/.well-known/agent.json", timeout=2)
                if response.status_code == 200:
                    email_ready = True
                    print("✅ Email Agent server ready")
            
            if cv_ready and email_ready:
                print("✅ All servers ready!")
                return True
                
        except:
            pass
        
        print(f"⏳ Attempt {attempt + 1}/30 - Waiting for servers...")
        time.sleep(1)
    
    print("❌ Servers not ready after 30 seconds")
    return False

def main():
    """Main test function"""
    print("🔧 OCR to Email Workflow Test")
    print("=" * 50)
    
    config = {
        "google_api_key": os.getenv("GOOGLE_API_KEY", "your_google_api_key_here"),
        "smtp_config": {
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "from_email": os.getenv("FROM_EMAIL", "your_email@gmail.com"),
            "username": os.getenv("SMTP_USERNAME", "your_email@gmail.com"),
            "password": os.getenv("SMTP_PASSWORD", "your_app_password")
        }
    }
    
    test_recipients = ["job2000@windowslive.com"]  
    
    test_image_path = "th.jpg"  
    if not os.path.exists(test_image_path):
        print(f"❌ Image not found: {test_image_path}")
        print("📝 Creating test image...")
        test_image_path = create_test_image()
        if not test_image_path:
            print("❌ Cannot create test image. Please provide a valid image file.")
            return
    
    if not wait_for_servers():
        print("❌ Please start the A2A servers first:")
        print("   Terminal 1: python test_cv.py")
        print("   Terminal 2: python test_email.py")
        return
    
    tester = OCREmailWorkflowTester(config)
    
    result = tester.run_full_workflow_test(test_image_path, test_recipients)
    
    print("\n📊 Final Test Results:")
    print("=" * 50)
    if result["status"] == "success":
        print("✅ Full workflow test PASSED!")
        print(f"📷 OCR processed successfully")
        print(f"📧 Email sent successfully")
    else:
        print("❌ Full workflow test FAILED!")
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    main()