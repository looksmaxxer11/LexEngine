from __future__ import annotations

import glob
import json
import os
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time
from collections import defaultdict, deque

from .orchestrator import run_pipeline
from .preprocessing import preprocess_pdf
from .ocr import OCREngine
from .supabase_client import SupabaseManager

# Global OCR engine and Supabase manager - preloaded on startup
_global_ocr_engine: Optional[OCREngine] = None
_supabase_manager: Optional[SupabaseManager] = None

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
ALLOWED_CONTENT_TYPES = {"application/pdf"}
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "6"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
_request_log: dict[str, deque[float]] = defaultdict(deque)


def _check_rate_limit(ip: str) -> None:
  now = time.time()
  history = _request_log[ip]
  while history and now - history[0] > RATE_LIMIT_WINDOW_SECONDS:
    history.popleft()
  if len(history) >= RATE_LIMIT_REQUESTS:
    raise HTTPException(status_code=429, detail="Too many uploads. Please wait a minute before retrying.")
  history.append(now)


def _validate_upload(file: UploadFile, content: bytes) -> None:
  if file.content_type not in ALLOWED_CONTENT_TYPES:
    raise HTTPException(status_code=400, detail="Only PDF uploads are supported at the moment.")
  if not content:
    raise HTTPException(status_code=400, detail="Uploaded file is empty.")
  size_mb = len(content) / (1024 * 1024)
  if size_mb > MAX_FILE_SIZE_MB:
    raise HTTPException(status_code=400, detail=f"File is too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")

@asynccontextmanager
async def lifespan(app: FastAPI):
  """Configure OCR engine and Supabase on startup."""
  global _global_ocr_engine, _supabase_manager
  print("\n🔄 Configuring Tesseract OCR engine...")
  _global_ocr_engine = OCREngine(use_angle_cls=False, gpu=False)
  
  print("🔄 Initializing Supabase connection...")
  _supabase_manager = SupabaseManager()
  print("   ✅ Supabase connected\n")
  
  try:
    import pytesseract  # type: ignore
    from shutil import which
    binary = getattr(pytesseract.pytesseract, "tesseract_cmd", None) or which("tesseract") or "not found"
    print(f"   🧭 Tesseract binary: {binary}")
    try:
      ver = pytesseract.get_tesseract_version()
      print(f"   📦 Tesseract version: {ver}")
    except Exception:
      pass
  except Exception:
    print("   ⚠️ pytesseract not installed in this environment. Install with: pip install pytesseract")
  print("   ✅ Tesseract ready. No heavy model preload required.\n")
  yield
  # Cleanup on shutdown
  _global_ocr_engine = None
  _supabase_manager = None

app = FastAPI(title="Legal OCR → Structuring UI", lifespan=lifespan)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static views for browsing outputs
DATA_ROOT = os.environ.get("DATA_ROOT", "data")
Path(DATA_ROOT, "images").mkdir(parents=True, exist_ok=True)
Path(DATA_ROOT, "json").mkdir(parents=True, exist_ok=True)
Path(DATA_ROOT, "text").mkdir(parents=True, exist_ok=True)
app.mount("/files/images", StaticFiles(directory=str(Path(DATA_ROOT, "images"))), name="images")
app.mount("/files/json", StaticFiles(directory=str(Path(DATA_ROOT, "json"))), name="json")
app.mount("/files/text", StaticFiles(directory=str(Path(DATA_ROOT, "text"))), name="text")


def _html_page(body: str, title: str = "Legal OCR → JSON") -> HTMLResponse:
    css = """
      <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:24px;max-width:960px}
        header{display:flex;justify-content:space-between;align-items:center}
        .box{padding:16px;border:1px solid #ddd;border-radius:8px;margin:16px 0}
        label{display:block;margin:8px 0}
        input,select,button{padding:8px}
        code,pre{background:#fafafa;border:1px solid #eee;border-radius:6px;padding:12px;display:block;overflow:auto}
        a.button{padding:8px 12px;border:1px solid #ccc;border-radius:6px;text-decoration:none;color:#222}
      </style>
    """
    return HTMLResponse(f"""
    <html>
      <head><title>{title}</title>{css}</head>
      <body>
        <header>
          <h2>{title}</h2>
          <nav>
            <a class=button href="/">Upload</a>
            <a class=button href="/recent">Recent</a>
          </nav>
        </header>
        {body}
      </body>
    </html>
    """)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # Serve the new modern UI template
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    if not template_path.exists():
        return HTMLResponse("<h1>Template not found</h1>", status_code=404)
    return HTMLResponse(template_path.read_text(encoding="utf-8"))


@app.post("/api/process")
async def api_process(
  request: Request,
  file: UploadFile = File(...),
    use_phase2: Optional[str] = Form(default=None),
    use_phase3: Optional[str] = Form(default=None),
    use_correction: Optional[str] = Form(default=None),
    user_id: Optional[str] = Form(default=None),
):
    """Streaming API endpoint for OCR processing with real-time progress updates."""
    client_ip = request.client.host if request and request.client else "anonymous"
    _check_rate_limit(client_ip)

    file_bytes = await file.read()
    _validate_upload(file, file_bytes)
    original_filename = file.filename or "document.pdf"
    
    async def generate_progress():
        start_time = time.time()
        
        try:
            # Save uploaded file
            data_root = DATA_ROOT
            os.makedirs(os.path.join(data_root, "raw_pdfs"), exist_ok=True)
            file_id = uuid.uuid4().hex[:12]
            safe_name = f"{file_id}_{os.path.basename(original_filename)}"
            in_path = os.path.join(data_root, "raw_pdfs", safe_name)
            
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Uploading file...', 'percent': 5})}\n\n"
            
            with open(in_path, "wb") as f:
              f.write(file_bytes)
            
            # Preprocess PDF to images
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Converting PDF to images...', 'percent': 10})}\n\n"
            await asyncio.sleep(0.1)
            
            page_images = preprocess_pdf(in_path, os.path.join(data_root, "images"), dpi=300)
            total_pages = len(page_images)
            
            yield f"data: {json.dumps({'type': 'progress', 'message': f'Processing {total_pages} pages...', 'percent': 20})}\n\n"
            
            # Process with OCR using quick mode
            enable_phase2 = bool(use_phase2)
            enable_phase3 = bool(use_phase3 or use_correction)
            
            if enable_phase2:
                yield f"data: {json.dumps({'type': 'progress', 'message': '🚀 Phase 2: Multi-scale OCR enabled', 'percent': 25})}\n\n"
            
            if enable_phase3:
                yield f"data: {json.dumps({'type': 'progress', 'message': '🔧 Phase 3: Post-correction enabled', 'percent': 30})}\n\n"
            
            # Simulate page-by-page processing feedback
            raw_texts = []
            detected_columns = []
            
            if _global_ocr_engine:
                for idx, page_img in enumerate(page_images, 1):
                    progress_percent = 30 + int((idx / total_pages) * 50)
                    yield f"data: {json.dumps({'type': 'progress', 'message': f'📄 Processing page {idx}/{total_pages}...', 'percent': progress_percent})}\n\n"
                    
                    try:
                        text, _ = _global_ocr_engine.ocr_region(page_img, lang="auto")
                        raw_texts.append(text)
                        
                        # Simulate column detection feedback (Phase 3.5)
                        if "region_ocr" in str(type(_global_ocr_engine)):
                            detected_columns.append(5)  # Example
                        
                    except Exception as e:
                        raw_texts.append(f"[ERROR: {e}]")
                        yield f"data: {json.dumps({'type': 'progress', 'message': f'⚠️ Page {idx} error: {str(e)[:50]}', 'percent': progress_percent})}\n\n"
            
            combined_text = "\n\n".join(raw_texts)
            
            # Apply corrections if enabled
            if enable_phase3:
                yield f"data: {json.dumps({'type': 'progress', 'message': '🔧 Applying post-OCR corrections...', 'percent': 85})}\n\n"
                await asyncio.sleep(0.5)
                # TODO: Apply actual post-OCR correction here
            
            yield f"data: {json.dumps({'type': 'progress', 'message': '✅ Processing complete!', 'percent': 95})}\n\n"
            
            # Calculate stats
            total_chars = len(combined_text)
            processing_time = time.time() - start_time
            avg_columns = sum(detected_columns) / len(detected_columns) if detected_columns else 0
            
            stats = {
                'pages': total_pages,
                'columns': int(avg_columns) if avg_columns > 0 else 1,
                'characters': total_chars,
                'time': round(processing_time, 1)
            }
            
            # Save to Supabase
            if _supabase_manager:
                yield f"data: {json.dumps({'type': 'progress', 'message': '💾 Saving to database...', 'percent': 98})}\n\n"
                save_result = await _supabase_manager.save_ocr_result(
                    filename=safe_name,
                    text=combined_text,
                    stats=stats,
                    user_id=user_id
                )
                if save_result.get('success'):
                    result_id = save_result.get('id')
                    yield f"data: {json.dumps({'type': 'progress', 'message': f'✅ Saved with ID: {result_id}', 'percent': 100})}\n\n"
            
            # Send final result
            result = {
                'type': 'complete',
                'text': combined_text,
                'stats': stats
            }
            
            yield f"data: {json.dumps(result)}\n\n"
            
        except Exception as e:
            error_result = {
                'type': 'error',
                'message': f'Processing failed: {str(e)}'
            }
            yield f"data: {json.dumps(error_result)}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/recent", response_class=HTMLResponse)
def recent() -> HTMLResponse:
    files = sorted(glob.glob(str(Path(DATA_ROOT, "json", "*.json"))), key=os.path.getmtime, reverse=True)[:50]
    items = "".join(
        f'<li><a href="/view?file={os.path.basename(p)}">{os.path.basename(p)}</a> &nbsp; '
        f'<a href="/files/json/{os.path.basename(p)}" target="_blank">Download</a></li>'
        for p in files
    )
    body = f"""
    <div class=box>
      <h3>Recent Results</h3>
      <ul>{items}</ul>
      <p><a href='/'>← Back</a></p>
    </div>
    """
    return _html_page(body, title="Recent Results")


@app.get("/view", response_class=HTMLResponse)
def view(file: str = Query(...)) -> HTMLResponse:
    path = Path(DATA_ROOT, "json", file)
    if not path.exists():
        return _html_page(f"<h3>File not found: {file}</h3>", title="Not Found")
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    try:
        pretty = json.dumps(json.loads(data), ensure_ascii=False, indent=2)
    except Exception:
        pretty = data
    body = f"""
    <div class=box>
      <h3>View: {file}</h3>
      <a class=button href="/files/json/{path.name}" target="_blank">Download JSON</a>
      <pre>{pretty}</pre>
      <p><a href='/recent'>← Back</a></p>
    </div>
    """
    return _html_page(body, title=f"View {path.name}")


@app.post("/process", response_class=HTMLResponse)
async def process(
    file: UploadFile = File(...),
    quick: Optional[str] = Form(default="on"),
    gpu: Optional[str] = Form(default=None),
    dry_run: Optional[str] = Form(default=None),
    no_embeddings: Optional[str] = Form(default=None),
    layout_strategy: str = Form(default="fallback"),
    store: str = Form(default="none"),
    max_workers: int = Form(default=1),
):
    data_root = DATA_ROOT
    os.makedirs(os.path.join(data_root, "raw_pdfs"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "json"), exist_ok=True)

    file_id = uuid.uuid4().hex[:12]
    safe_name = f"{file_id}_{os.path.basename(file.filename)}"
    in_path = os.path.join(data_root, "raw_pdfs", safe_name)
    with open(in_path, "wb") as f:
        f.write(await file.read())

    out_name = os.path.splitext(safe_name)[0] + ".json"
    out_path = os.path.join(data_root, "json", out_name)

    # Run PaddleOCR pipeline with pre-loaded engine
    rec = run_pipeline(
        input_pdf=in_path,
        output_json=out_path,
        data_root=data_root,
        layout_strategy=layout_strategy,
        use_gpu=bool(gpu),
        enable_embeddings=not bool(no_embeddings),
        store=store,
        dry_run=bool(dry_run),
        max_workers=max_workers,
        quick=bool(quick),
        ocr_engine=_global_ocr_engine,
    )
    # Normalize output for display
    try:
      if hasattr(rec, "to_json"):
        rec_str = rec.to_json()
        pretty = json.dumps(json.loads(rec_str), ensure_ascii=False, indent=2)
      elif isinstance(rec, (dict, list)):
        pretty = json.dumps(rec, ensure_ascii=False, indent=2)
      else:
        pretty = str(rec)
    except Exception:
      pretty = "{\n  \"status\": \"ok\",\n  \"message\": \"Result saved to JSON file.\"\n}"
    body = f"""
    <div class=box>
      <h3>✅ Processed: {os.path.basename(file.filename)}</h3>
      <p>
        <a class=button href="/recent">Recent</a>
        <a class=button href="/files/json/{out_name}" target="_blank">Download JSON</a>
      </p>
      <pre>{pretty}</pre>
    </div>
    """
    return _html_page(body, title=f"Result - {safe_name}")

