from fastapi import FastAPI, HTTPException, Depends, status, Request, Header
from fastapi.security import HTTPBasic, HTTPBasicCredentials, APIKeyHeader
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import secrets
import os
from huggingface_hub import snapshot_download
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

# Configuration
USERNAME = "admin"
PASSWORD = "Gaurav"
API_KEY = os.environ.get("API_KEY", "Gaurav@08")  


api_key_header = APIKeyHeader(name="X-API-Key")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = None
tokenizer = None

os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

with open("templates/index.html", "w") as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text Classification API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            textarea { width: 100%; height: 100px; margin-bottom: 10px; }
            input { width: 100%; margin-bottom: 10px; padding: 8px; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            #result { margin-top: 20px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>Text Classification API</h1>
        <input type="password" id="apiKey" placeholder="Enter your API key">
        <textarea id="prompt" placeholder="Enter text to classify..."></textarea>
        <div>
            <button onclick="classifyText()">Classify</button>
        </div>
        <div id="result"></div>

        <script>
            async function classifyText() {
                const prompt = document.getElementById('prompt').value;
                const apiKey = document.getElementById('apiKey').value;
                const result = document.getElementById('result');
                
                if (!prompt) {
                    result.textContent = "Please enter some text to classify";
                    return;
                }
                
                if (!apiKey) {
                    result.textContent = "Please enter your API key";
                    return;
                }
                
                result.textContent = "Classifying...";
                
                try {
                    const response = await fetch('/classify/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-API-Key': apiKey
                        },
                        body: JSON.stringify({ text: prompt })
                    });
                    
                    const data = await response.json();
                    if (response.ok) {
                        result.textContent = JSON.stringify(data, null, 2);
                    } else {
                        result.textContent = "Error: " + data.detail;
                    }
                } catch (error) {
                    result.textContent = "Error: " + error.message;
                }
            }
        </script>
    </body>
    </html>
    """)

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return api_key

def download_and_load_model():
    global model, tokenizer
    
    logger.info(f"Downloading model {model_name}...")
    
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    
    try:
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=False,
        )
        
        logger.info(f"Model downloaded to {model_path}")
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info("Loading model...")
        try:
            # First try with device_map if accelerate is available
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None
            )
        except ImportError:
            # Fall back to simpler loading method if accelerate isn't available
            logger.info("Accelerate not available, loading model without device_map...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            )

class ClassificationRequest(BaseModel):
    text: str

@app.post("/classify/")
async def classify_text(request: ClassificationRequest, api_key: str = Depends(verify_api_key)):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
        
    try:
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        scores = predictions[0].tolist()
        result = {"sentiment": "positive" if scores[1] > scores[0] else "negative", 
                  "confidence": max(scores)}
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    status_msg = "ok" if model is not None else "loading"
    return {"status": status_msg, "model": model_name}

@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=download_and_load_model)
    thread.daemon = True
    thread.start()

# For RunPod, we need to listen on 0.0.0.0 and port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)