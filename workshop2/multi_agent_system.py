import json
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import List, Dict, Any, Optional
from datetime import datetime
import base64
import io
import os
import logging
from PIL import Image, ImageDraw, ImageFont
import uvicorn


from google.adk.agents import Agent


from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from a2a.utils import new_agent_text_message


from google import genai
from google.genai import types


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_thai_font(size: int = 24) -> ImageFont.FreeTypeFont:
    """
    Attempts to find and load a Thai-compatible font from the system.
    Falls back to a default font if no specific Thai font is found.
    """
    # Common font paths for Linux, macOS, and Windows
    font_paths = [
        # Linux (Debian/Ubuntu)
        "/usr/share/fonts/truetype/thai/Norasi.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Thonburi.ttf",
        # Windows
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/Leelawadee.ttf",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                logger.info(f"Loading Thai font from: {font_path}")
                return ImageFont.truetype(font_path, size)
            except Exception as e:
                logger.warning(f"Could not load font {font_path}: {e}")

    logger.warning("No Thai font found on the system. Falling back to default font. Thai characters may not render correctly.")
    return ImageFont.load_default()


# --- Bounding box drawing function ---
def draw_bounding_boxes_with_labels(image_path: str, detections: List[Dict[str, Any]]) -> Optional[bytes]:
    """
    Draw bounding boxes and Thai labels on the image using Pillow.
    This function now also saves the annotated image to the local disk.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = _get_thai_font(size=max(15, int(image.height / 50))) # Dynamic font size

        w, h = image.size
        
        for detection in detections:
            box = detection.get('box_2d')
            label = detection.get('label')

            # Ensure box contains valid coordinate data
            if not (box and label and isinstance(box, list) and len(box) == 4):
                continue

            # Denormalize coordinates
            y1, x1, y2, x2 = [
                int(coord / 1000 * (h if i % 2 == 0 else w)) 
                for i, coord in enumerate(box)
            ]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            
            # Prepare text label
            try:
                # Get text size using textbbox for modern Pillow
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                # Fallback for older Pillow versions
                text_width, text_height = draw.textsize(label, font=font)

            # Position label and draw background
            label_y_pos = max(0, y1 - text_height - 5)
            background_box = [x1, label_y_pos, x1 + text_width + 4, label_y_pos + text_height + 4]
            draw.rectangle(background_box, fill="lime")
            
            # Draw text
            draw.text((x1 + 2, label_y_pos + 2), label, fill="black", font=font)

        # Save the annotated image to disk
        base, ext = os.path.splitext(image_path)
        annotated_image_path = f"{base}_annotated.png"
        image.save(annotated_image_path)
        logger.info(f"💾 Annotated image saved locally to: {annotated_image_path}")

        # Save image to a byte buffer for emailing
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    except Exception as e:
        logger.error(f"Error drawing bounding boxes with Pillow: {e}")
        return None

# --- *** THE FIX IS IN THIS FUNCTION *** ---
def process_ocr_image(image_path: str, google_api_key: str) -> Dict[str, Any]:
    """
    ADK Tool Function: Extract text from images including Thai text using Google Gemini Vision
    
    Args:
        image_path: Path to the image file
        google_api_key: Google API key for Gemini
        
    Returns:
        Dictionary containing OCR results
    """
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=google_api_key)
        
        prompt = "Extract all text from this image, including Thai text. For each detected text region, provide the bounding box coordinates and the text content."
        output_prompt = """
        Return the results in JSON format with the following structure:
        [{"box_2d": [y1, x1, y2, x2], "label": "extracted_text_here"}]
        where coordinates are normalized from 0 to 1000. Return only the JSON array.
        """
        
        image = Image.open(image_path)
        
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt + output_prompt, image],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048
            )
        )
        
        # --- NEW ROBUST HANDLING FOR API RESPONSE ---
        
        # The .text property raises ValueError if the response is blocked
        try:
            text_response = response.text
        except ValueError:
            # Safely access the block reason
            block_reason = "Unknown"
            if response.prompt_feedback:
                block_reason = response.prompt_feedback.block_reason.name
            error_msg = f"Gemini API response was blocked. Reason: {block_reason}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        # Handle cases where the response is not blocked but still empty
        if text_response is None:
            error_msg = "Gemini API returned an empty (None) response."
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
            
        # --- END OF NEW HANDLING ---

        # Proceed with cleaning and parsing if the response is valid
        cleaned_results = _clean_gemini_results(text_response)
        detections = json.loads(cleaned_results)
        
        return {
            "status": "success",
            "detections": detections,
            "extracted_text": [item["label"] for item in detections],
            "total_regions": len(detections)
        }
        
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def send_ocr_email(recipients: List[str], subject: str, 
                   ocr_results: Dict[str, Any], attachments: List[str] = None,
                   smtp_config: Dict[str, str] = None, image_path: str = None) -> Dict[str, Any]:
    """
    ADK Tool Function: Send emails with OCR results and an annotated image.
    
    Args:
        recipients: List of email addresses
        subject: Email subject
        ocr_results: OCR processing results
        attachments: List of other attachment file paths
        smtp_config: SMTP configuration
        image_path: Path to the original image to be annotated
        
    Returns:
        Dictionary containing email sending results
    """
    try:
        if not smtp_config:
            return {"status": "error", "error": "SMTP configuration not provided"}

        # Create message
        msg = MIMEMultipart('related')
        msg['From'] = smtp_config['from_email']
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject

        # Draw bounding boxes on the image (this function now also saves the file)
        annotated_image_data = None
        if image_path and os.path.exists(image_path):
            annotated_image_data = draw_bounding_boxes_with_labels(image_path, ocr_results.get('detections', []))

        # Create email body
        html_body = _format_ocr_email_body(ocr_results, has_annotated_image=annotated_image_data is not None)
        msg.attach(MIMEText(html_body, 'html'))

        # Attach the annotated image to the email
        if annotated_image_data:
            img = MIMEImage(annotated_image_data, name=f"annotated_{os.path.basename(image_path)}")
            img.add_header('Content-ID', '<annotated_image>')
            img.add_header('Content-Disposition', 'inline', filename=img.get_filename())
            msg.attach(img)
        
        # Add other attachments if provided
        if attachments:
            for attachment_path in attachments:
                if os.path.exists(attachment_path):
                    with open(attachment_path, 'rb') as f:
                        attachment = MIMEImage(f.read())
                        attachment.add_header('Content-Disposition', 
                                           f'attachment; filename={os.path.basename(attachment_path)}')
                        msg.attach(attachment)
        
        # Send email
        with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
        
        return {
            "status": "success",
            "message_id": f"email_{datetime.now().isoformat()}",
            "recipients": recipients,
            "text_regions_count": len(ocr_results.get('extracted_text', []))
        }
        
    except Exception as e:
        logger.error(f"Email sending error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def _clean_gemini_results(results: str) -> str:
    """Clean the results for JSON parsing"""
    cleaned = results.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def _format_ocr_email_body(ocr_results: Dict[str, Any], has_annotated_image: bool = False) -> str:
    """Format OCR results into HTML email body"""
    extracted_text = ocr_results.get('extracted_text', [])
    detections = ocr_results.get('detections', [])
    
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
            .text-region {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #007cba; }}
            .coordinates {{ font-size: 12px; color: #666; }}
            .summary {{ background-color: #e6f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .annotated-image {{ border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>OCR Processing Results</h2>
            <p>Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """

    if has_annotated_image:
        html_body += """
        <div class="annotated-image">
            <h3>Annotated Image</h3>
            <p>The image below shows the locations of the detected text. A copy of this image has also been saved to your local directory.</p>
            <img src="cid:annotated_image" alt="Annotated OCR Image" style="max-width: 100%; height: auto;">
        </div>
        """
        
    html_body += f"""
        <div class="summary">
            <h3>Summary</h3>
            <p><strong>Total text regions detected:</strong> {len(extracted_text)}</p>
            <p><strong>Languages detected:</strong> Thai, English</p>
        </div>
        
        <h3>Extracted Text Regions</h3>
    """
    
    for i, (text, detection) in enumerate(zip(extracted_text, detections), 1):
        html_body += f"""
        <div class="text-region">
            <h4>Region {i}</h4>
            <p><strong>Text:</strong> {text}</p>
            <p class="coordinates"><strong>Coordinates:</strong> {detection.get('box_2d', 'N/A')}</p>
        </div>
        """
    
    html_body += """
        <div class="header">
            <p><em>This email was generated automatically by the Multi-Agent OCR System</em></p>
        </div>
    </body>
    </html>
    """
    
    return html_body

# A2A Agent Executors
class ComputerVisionAgentExecutor(AgentExecutor):
    """A2A Agent Executor for Computer Vision tasks"""
    
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        try:
            genai.configure(api_key=self.google_api_key)
            logger.info("✅ Google GenAI configured successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to configure Google GenAI: {e}")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Execute OCR processing task and return the final result directly.
        """
        try:
            image_path = self._extract_image_path_from_context(context)
            if not image_path:
                await event_queue.enqueue_event(new_agent_text_message(
                    "Error: No image path provided."
                ))
                return

            # Run the blocking function in a separate thread
            result = await asyncio.to_thread(
                process_ocr_image, image_path, self.google_api_key
            )

            # Instead of sending a formatted string, send the full JSON result.
            if result["status"] == "success":
                # Convert the entire result dictionary to a JSON string
                json_result = json.dumps(result, ensure_ascii=False)
                await event_queue.enqueue_event(new_agent_text_message(json_result))
            else:
                # For errors, we can still send a formatted string
                error_message = f"❌ OCR Processing Failed: {result.get('error', 'Unknown error')}"
                await event_queue.enqueue_event(new_agent_text_message(error_message))

        except Exception as e:
            logger.error(f"OCR execution error: {e}")
            await event_queue.enqueue_event(new_agent_text_message(
                f"❌ Unexpected error during OCR processing: {str(e)}"
            ))
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel OCR processing task"""
        await event_queue.enqueue_event(new_agent_text_message(
            "⏹️ OCR processing task cancelled."
        ))
    
    def _extract_image_path_from_context(self, context: RequestContext) -> Optional[str]:
        """Extract image path from RequestContext"""
        try:
            # Try to get user input using the context method
            user_input = context.get_user_input()
            if user_input and "image_path:" in user_input:
                return user_input.split("image_path:")[1].strip()
            
            # Fallback: check message parts
            if context.message and hasattr(context.message, 'parts'):
                for part in context.message.parts:
                    if hasattr(part, 'text') and part.text:
                        if "image_path:" in part.text:
                            return part.text.split("image_path:")[1].strip()
                    elif hasattr(part, 'root') and hasattr(part.root, 'text'):
                        if "image_path:" in part.root.text:
                            return part.root.text.split("image_path:")[1].strip()
            
            return None
        except Exception as e:
            logger.error(f"Error extracting image path: {e}")
            return None

class EmailAgentExecutor(AgentExecutor):
    """A2A Agent Executor for Email tasks"""
    
    def __init__(self, smtp_config: Dict[str, str]):
        self.smtp_config = smtp_config
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute email sending task and return the result directly."""
        try:
            email_request = self._parse_email_request_from_context(context)
            
            if not email_request:
                await event_queue.enqueue_event(new_agent_text_message(
                    "Error: Invalid email request format."
                ))
                return
            
            # FIX: Run the blocking email function in a separate thread
            result = await asyncio.to_thread(
                send_ocr_email,
                recipients=email_request['recipients'],
                subject=email_request['subject'],
                ocr_results=email_request['ocr_results'],
                attachments=email_request.get('attachments', []),
                smtp_config=self.smtp_config,
                image_path=email_request.get('image_path')
            )

            if result["status"] == "success":
                response_text = f"""
✅ Email Sent Successfully!

📬 **Email Details:**
- Recipients: {', '.join(result['recipients'])}
- Text regions included: {result['text_regions_count']}
                """
                await event_queue.enqueue_event(new_agent_text_message(response_text))
            else:
                await event_queue.enqueue_event(new_agent_text_message(
                    f"❌ Email Sending Failed: {result['error']}"
                ))
                
        except Exception as e:
            logger.error(f"Email execution error: {e}")
            await event_queue.enqueue_event(new_agent_text_message(
                f"❌ Unexpected error during email sending: {str(e)}"
            ))
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel email sending task"""
        await event_queue.enqueue_event(new_agent_text_message(
            "⏹️ Email sending task cancelled."
        ))
    
    def _parse_email_request_from_context(self, context: RequestContext) -> Optional[Dict[str, Any]]:
        """Parse email request from RequestContext"""
        try:
            # Try to get user input using the context method
            user_input = context.get_user_input()
            if user_input:
                return json.loads(user_input)
            
            # Fallback: check message parts
            if context.message and hasattr(context.message, 'parts'):
                for part in context.message.parts:
                    if hasattr(part, 'text') and part.text:
                        return json.loads(part.text)
                    elif hasattr(part, 'root') and hasattr(part.root, 'text'):
                        return json.loads(part.root.text)
            
            return None
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error parsing email request: {e}")
            return None

# ADK Agents
class ComputerVisionADKAgent:
    """ADK Agent for Computer Vision tasks"""
    
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        
        # Create OCR processing function with API key bound
        def ocr_tool(image_path: str) -> Dict[str, Any]:
            """Process OCR on the given image path"""
            return process_ocr_image(image_path, self.google_api_key)
        
        # Create ADK Agent
        self.agent = Agent(
            name="cv_agent",
            model="gemini-2.0-flash",
            description="Computer Vision agent that processes images and extracts text using OCR",
            instruction="""
            You are a computer vision agent specialized in OCR (Optical Character Recognition).
            Your main capability is to extract text from images, including Thai text.
            
            When a user provides an image path, use the ocr_tool to process it.
            Always provide detailed results including:
            - Number of text regions detected
            - Extracted text content
            - Bounding box coordinates when available
            
            Be helpful and explain what was found in the image.
            """,
            tools=[ocr_tool]
        )

class EmailADKAgent:
    """ADK Agent for Email tasks"""
    
    def __init__(self, smtp_config: Dict[str, str]):
        self.smtp_config = smtp_config
        
        # Create email sending function with SMTP config bound
        def email_tool(recipients: List[str], subject: str, ocr_results: Dict[str, Any], 
                      attachments: List[str] = None, image_path: str = None) -> Dict[str, Any]:
            """Send email with OCR results"""
            return send_ocr_email(recipients, subject, ocr_results, attachments, self.smtp_config, image_path)
        
        # Create ADK Agent
        self.agent = Agent(
            name="email_agent",
            model="gemini-2.0-flash",
            description="Email agent that sends OCR results via email",
            instruction="""
            You are an email agent specialized in sending OCR processing results.
            
            When requested to send an email, use the email_tool with the following parameters:
            - recipients: List of email addresses
            - subject: Email subject line
            - ocr_results: The OCR processing results to include in the email
            - attachments: Optional list of file paths to attach
            - image_path: The path to the original image to include with bounding boxes.
            
            Always confirm successful email delivery and provide details about what was sent.
            """,
            tools=[email_tool]
        )

# A2A Server Factory Functions
def create_cv_agent_server(google_api_key: str, host: str = "localhost", port: int = 5001):
    """Create A2A server for Computer Vision agent"""
    
    # Define OCR processing skill
    ocr_skill = AgentSkill(
        id='ocr_processing',
        name='OCR Text Extraction',
        description='Extract text from images including Thai characters using Google Gemini Vision',
        tags=['ocr', 'text extraction', 'thai', 'vision'],
        examples=['Extract text from this image', 'Process OCR on image_path: /path/to/image.jpg'],
    )
    
    # Define agent card
    agent_card = AgentCard(
        name='Computer Vision Agent',
        description='Processes images and extracts text using OCR with Thai language support',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[ocr_skill],
    )
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=ComputerVisionAgentExecutor(google_api_key),
        task_store=InMemoryTaskStore(),
    )
    
    # Create A2A server
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    return server, host, port

def create_email_agent_server(smtp_config: Dict[str, str], host: str = "localhost", port: int = 5002):
    """Create A2A server for Email agent"""
    
    # Define email sending skill
    email_skill = AgentSkill(
        id='email_sending',
        name='Email Sending',
        description='Send emails with OCR results and attachments',
        tags=['email', 'smtp', 'ocr results', 'attachments'],
        examples=['Send email with OCR results', 'Email results to recipients'],
    )
    
    # Define agent card
    agent_card = AgentCard(
        name='Email Agent',
        description='Sends emails with OCR processing results and attachments',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[email_skill],
    )
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=EmailAgentExecutor(smtp_config),
        task_store=InMemoryTaskStore(),
    )
    
    # Create A2A server
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    return server, host, port

class MultiAgentOCRSystem:
    """Main system coordinating ADK agents with A2A protocol"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adk_agents = {}
        self.a2a_servers = {}
        
    async def initialize_agents(self):
        """Initialize all ADK agents and A2A servers"""
        
        # 1. Initialize ADK agents
        cv_adk_agent = ComputerVisionADKAgent(self.config["google_api_key"])
        email_adk_agent = EmailADKAgent(self.config["smtp_config"])
        
        self.adk_agents = {
            "cv_agent": cv_adk_agent,
            "email_agent": email_adk_agent,
        }
        
        # 2. Initialize A2A servers
        cv_server, cv_host, cv_port = create_cv_agent_server(self.config["google_api_key"])
        email_server, email_host, email_port = create_email_agent_server(self.config["smtp_config"])
        
        self.a2a_servers = {
            "cv_server": (cv_server, cv_host, cv_port),
            "email_server": (email_server, email_host, email_port),
        }
        
        logger.info("All agents initialized successfully")
    
    async def start_a2a_servers(self):
        """Start A2A servers"""
        if not self.a2a_servers:
            await self.initialize_agents()
        
        server_tasks = []
        for server_name, (server, host, port) in self.a2a_servers.items():
            task = asyncio.create_task(self._run_server(server, host, port, server_name))
            server_tasks.append(task)
        
        logger.info("All A2A servers started")
        return server_tasks
    
    async def _run_server(self, server, host: str, port: int, server_name: str):
        """Run individual A2A server"""
        try:
            logger.info(f"Starting {server_name} on {host}:{port}")
            config = uvicorn.Config(
                app=server.build(),
                host=host,
                port=port,
                log_level="info"
            )
            server_instance = uvicorn.Server(config)
            await server_instance.serve()
        except Exception as e:
            logger.error(f"Error starting {server_name}: {e}")
    
    def get_cv_agent(self) -> Agent:
        """Get the CV ADK agent"""
        return self.adk_agents["cv_agent"].agent
    
    def get_email_agent(self) -> Agent:
        """Get the Email ADK agent"""
        return self.adk_agents["email_agent"].agent
    
    def run_cv_server(self):
        """Run CV agent A2A server"""
        server, host, port = self.a2a_servers["cv_server"]
        uvicorn.run(server.build(), host=host, port=port)
    
    def run_email_server(self):
        """Run Email agent A2A server"""
        server, host, port = self.a2a_servers["email_server"]
        uvicorn.run(server.build(), host=host, port=port)

# Individual server runner functions
def run_cv_agent_server(google_api_key: str):
    """Run CV agent A2A server standalone"""
    # Ensure API key is available
    if not google_api_key or google_api_key == "your_google_api_key_here":
        print("❌ Error: Google API key not provided!")
        print("Please set GOOGLE_API_KEY environment variable or pass it as argument")
        return
    
    server, host, port = create_cv_agent_server(google_api_key)
    print(f"🚀 Starting Computer Vision Agent server on {host}:{port}")
    print(f"📍 Agent card available at: http://{host}:{port}/.well-known/agent.json")
    print(f"🔑 Using API key: {google_api_key[:10]}...{google_api_key[-5:]}")
    uvicorn.run(server.build(), host=host, port=port)

def run_email_agent_server(smtp_config: Dict[str, str]):
    """Run Email agent A2A server standalone"""
    server, host, port = create_email_agent_server(smtp_config)
    print(f"🚀 Starting Email Agent server on {host}:{port}")
    print(f"📍 Agent card available at: http://{host}:{port}/.well-known/agent.json")
    uvicorn.run(server.build(), host=host, port=port)

# Example usage and testing
async def main():
    """Example usage of the multi-agent system with Google ADK and A2A"""
    
    # Configuration
    config = {
        "google_api_key": "your_google_api_key_here",
        "smtp_config": {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "from_email": "your_email@gmail.com",
            "username": "your_email@gmail.com",
            "password": "your_app_password"
        }
    }
    
    # Create system
    system = MultiAgentOCRSystem(config)
    
    try:
        # Initialize agents
        await system.initialize_agents()
        
        # Test individual agents
        print("🚀 Testing individual ADK agents...")
        
        # Test CV agent
        cv_agent = system.get_cv_agent()
        print("📷 CV Agent ready")
        
        # Test Email agent
        email_agent = system.get_email_agent()
        print("📧 Email Agent ready")
        
        print("✅ All agents are ready!")
        print("\n🔧 To run A2A servers, use these commands:")
        print("   python -c \"from multi_agent_system import run_cv_agent_server; run_cv_agent_server('your_api_key')\"")
        print("   python -c \"from multi_agent_system import run_email_agent_server; run_email_agent_server({'smtp_server': 'smtp.gmail.com', 'smtp_port': 587, 'from_email': 'your_email@gmail.com', 'username': 'your_email@gmail.com', 'password': 'your_password'})\"")
        
        print("\n💡 Or you can interact with ADK agents using:")
        print("   adk web (to start web interface)")
        print("   adk run cv_agent (to run CV agent)")
        print("   adk run email_agent (to run Email agent)")
        
    except Exception as e:
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    # Installation instructions
    print("""
    🔧 Installation Instructions:
    
    1. Install required packages:
       pip install google-adk a2a-sdk google-genai pillow uvicorn python-dotenv
    
    2. Set up environment variables:
       export GOOGLE_API_KEY="your_google_api_key"
       export GOOGLE_CLOUD_PROJECT="your_project_id"
    
    3. Configure SMTP settings in the config dictionary or a .env file.
    
    4. Choose your usage mode:
       
       A) Run individual A2A servers:
          python test_cv.py
          python test_email.py
       
       B) Run the full workflow test:
          python test_full_workflow.py
    
    5. FONT REQUIREMENT:
       For Thai labels to appear correctly on the image, you must have a Thai-compatible
       TrueType font installed. This script checks for common fonts on Linux, macOS,
       and Windows. If none are found, labels will use a default font and may not
       render correctly.
    
    📚 Documentation:
    - ADK: https://google.github.io/adk-docs/
    - A2A: https://a2aprotocol.ai/
    """)
    
    # Run the system
    asyncio.run(main())