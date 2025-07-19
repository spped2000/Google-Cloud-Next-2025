import os
import uvicorn
import aiofiles
from fastapi import FastAPI, Request, File, UploadFile, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


try:
    from test_full_workflow import OCREmailWorkflowTester, wait_for_servers
except ImportError:
    print("❌ Error: Make sure 'test_full_workflow.py' is in the same directory.")
    exit(1)


app = FastAPI()

os.makedirs("uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="uploads"), name="static")

templates = Jinja2Templates(directory="templates")

# --- Main Application Logic ---

def get_tester():
    config = {"google_api_key": os.getenv("GOOGLE_API_KEY"), "smtp_config": {}}
    return OCREmailWorkflowTester(config)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the initial upload form."""
    return templates.TemplateResponse("template.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
async def process_image(
    request: Request,
    recipient: str = Form(...),
    image: UploadFile = File(...),
    tester: OCREmailWorkflowTester = Depends(get_tester)
):
    """Handles image upload, orchestration, and displays results."""
    results = {}
    error = None

    try:
        file_path = os.path.join("uploads", image.filename)
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await image.read()
            await out_file.write(content)

        # 2. Run the existing synchronous workflow in a separate thread
        # This is crucial to avoid blocking the async event loop
        import asyncio
        results = await asyncio.to_thread(
            tester.run_full_workflow_test, file_path, [recipient]
        )

        if results.get("status") != "success":
            error = results.get('error', 'An unknown error occurred in the workflow.')

    except Exception as e:
        error = f"An unexpected error occurred: {e}"

    # 3. Render the template again with the results
    return templates.TemplateResponse("template.html", {
        "request": request,
        "results": results,
        "error": error,
        "processed_image_filename": image.filename # To display the annotated image
    })


# --- Main execution block ---
if __name__ == "__main__":
    print("🔧 Initializing FastAPI UI for OCR Workflow...")

    # Wait for the backend agent servers to be ready
    if not wait_for_servers():
        print("\n❌ Could not connect to backend agent servers.")
        print("   Please start them first in separate terminals:")
        print("   Terminal 1: python test_cv.py")
        print("   Terminal 2: python test_email.py")
    else:
        print("\n✅ Backend agents are running.")
        print("🚀 Starting FastAPI server with Uvicorn...")
        print("🌍 Open your browser and go to: http://127.0.0.1:8000")
        print("📚 API documentation available at: http://127.0.0.1:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)
