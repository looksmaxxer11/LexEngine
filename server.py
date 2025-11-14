"""
Minimal FastAPI server wrapping the local OCR pipeline.
Run: uvicorn server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import os
import platform

from src.pipeline import run_pipeline
import re
from typing import Optional, List
import urllib.request
import pathlib


class ProcessRequest(BaseModel):
    path: str


app = FastAPI(title="Local OCR Pipeline")

# Mount static files for UI
app.mount("/static", StaticFiles(directory="static"), name="static")


def _auto_configure_tesseract() -> str:
    """Attempt to locate the Tesseract executable on Windows and set pytesseract path.

    Order of resolution:
    1. Environment variable `TESSERACT_CMD`
    2. Common install paths (64-bit / 32-bit)
    3. Existing configured path (if already set)
    Returns the resolved path or a placeholder string.
    """
    try:
        import pytesseract
        from pytesseract import pytesseract as pyt
    except Exception:
        return "pytesseract-not-installed"

    # If already configured externally, respect it
    current = getattr(pyt, "tesseract_cmd", "tesseract")
    if os.path.isabs(current) and os.path.exists(current):
        return current

    # 1. Environment override
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.exists(env_path):
        pyt.tesseract_cmd = env_path
        return env_path

    # 2. Common Windows install locations
    if platform.system().lower() == "windows":
        candidates = [
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                pyt.tesseract_cmd = c
                return c

    # Leave as-is (will rely on PATH)
    return current


def _download_tess_language(lang: str, tessdata_dir: str) -> bool:
    """Download a Tesseract language traineddata file into tessdata_dir.
    Returns True if downloaded successfully or already exists, False otherwise."""
    try:
        os.makedirs(tessdata_dir, exist_ok=True)
        target_path = os.path.join(tessdata_dir, f"{lang}.traineddata")
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return True
        url = f"https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang}.traineddata"
        print(f"⬇️  Downloading {lang} language data from {url} ...")
        urllib.request.urlretrieve(url, target_path)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"✅ Installed {lang}.traineddata to {tessdata_dir}")
            return True
        print(f"⚠️  Downloaded {lang}.traineddata appears invalid (zero size).")
        return False
    except Exception as e:
        print(f"⚠️  Failed to download language {lang}: {e}")
        return False


def _ensure_local_tessdata(required: List[str]) -> str:
    """Ensure a writable, project‑local tessdata directory containing the required languages.

    Strategy:
    1. Use ./tessdata (created if missing) and set TESSDATA_PREFIX to it so pytesseract uses it.
    2. For each required language (plus 'osd' for orientation detection):
       a) If already present -> skip.
       b) Attempt to copy from system tessdata (queried via 'tesseract --print-tessdata-dir').
       c) Fallback: try common Windows install path.
       d) If still missing -> download from tessdata_best.
    Returns the absolute path to the local tessdata directory.
    """
    import shutil, subprocess

    local_dir = Path("tessdata").resolve()
    os.makedirs(local_dir, exist_ok=True)
    os.environ["TESSDATA_PREFIX"] = str(local_dir)

    # Attempt to discover system tessdata directory via tesseract command
    system_tessdata: Optional[Path] = None
    try:
        out = subprocess.check_output(["tesseract", "--print-tessdata-dir"], stderr=subprocess.STDOUT, timeout=5)
        cand = Path(out.decode("utf-8", errors="ignore").strip())
        if cand.exists():
            system_tessdata = cand
    except Exception:
        pass

    # Common Windows paths fallback
    if system_tessdata is None and platform.system().lower() == "windows":
        for p in [
            Path(r"C:\\Program Files\\Tesseract-OCR\\tessdata"),
            Path(r"C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"),
        ]:
            if p.exists():
                system_tessdata = p
                break

    all_required = list(dict.fromkeys(required + ["osd"]))  # keep order, add OSD
    for lang in all_required:
        target = local_dir / f"{lang}.traineddata"
        if target.exists() and target.stat().st_size > 0:
            continue

        copied = False
        if system_tessdata:
            source = system_tessdata / f"{lang}.traineddata"
            if source.exists() and source.stat().st_size > 0:
                try:
                    shutil.copyfile(source, target)
                    print(f"📄 Copied {lang}.traineddata from system tessdata to local directory")
                    copied = True
                except Exception as e:
                    print(f"⚠️  Failed copying {lang} from system tessdata: {e}")

        if not copied:
            ok = _download_tess_language(lang, str(local_dir))
            if not ok:
                print(f"❌ Unable to provision language '{lang}'. OCR accuracy may degrade.")

    # Final listing
    available = sorted([p.stem for p in local_dir.glob("*.traineddata") if p.stat().st_size > 0])
    print(f"🔤 Local tessdata ready at {local_dir}. Languages: {', '.join(available) if available else '(none)'}")
    return str(local_dir)


@app.on_event("startup")
async def startup_info():
    """Configure and log Tesseract details at startup."""
    resolved = _auto_configure_tesseract()
    try:
        import pytesseract
        from pytesseract import pytesseract as pyt
        # Prepare local tessdata BEFORE enumerating languages so detection uses it
        local_tess = _ensure_local_tessdata(["eng", "rus", "uzb"])
        ver = pytesseract.get_tesseract_version()
        print(f"🧭 Tesseract binary: {getattr(pyt, 'tesseract_cmd', resolved)}\n📦 Tesseract version: {ver}\n📁 Using local tessdata: {local_tess}")
        if "pytesseract-not-installed" == resolved:
            print("❌ pytesseract not installed. Install with: pip install pytesseract")
        if not os.path.isabs(resolved):
            print("ℹ️  Using system PATH lookup. Set TESSERACT_CMD env var for explicit path if needed.")
        try:
            langs = pytesseract.get_languages(config="")  # should reflect local tessdata now
        except Exception:
            langs = []
        if langs:
            print(f"🔤 Installed languages: {', '.join(sorted(langs))}")
            # No extra download logic needed; _ensure_local_tessdata already handled provisioning.
        else:
            print("ℹ️  Could not enumerate installed Tesseract languages. Ensure tessdata directory is accessible.")
    except Exception as e:
        print(f"⚠️  Tesseract check failed: {e}\n👉 If not installed on Windows, download the installer: https://github.com/UB-Mannheim/tesseract/wiki")


@app.get("/")
async def root():
    """Redirect to UI."""
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
def process(req: ProcessRequest):
    pdf = Path(req.path)
    if not pdf.exists() or pdf.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Invalid or missing PDF path")
    try:
        result = run_pipeline(
            input_pdf=str(pdf),
            output_json=f"data/json/{pdf.stem}.json",
            data_root="data",
            max_workers=1,
            enable_embeddings=False  # disable heavy embedding model for faster processing
        )
        return {
            "ok": True,
            "ref": result.ref,
            "date": result.date,
            "language": result.language,
            "json_saved": f"data/json/{pdf.stem}.json",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _normalize_path(p: str | None) -> str | None:
    if not p:
        return None
    p = p.strip().strip('"')
    # Collapse duplicate backslashes
    p = re.sub(r"\\{2,}", r"\\", p)
    return os.path.normpath(p)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler to avoid UTF-8 decoding of raw PDF bytes in validation errors."""
    def _sanitize(obj):
        if isinstance(obj, bytes):
            return f"<{len(obj)} bytes>"
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    raw_errors = exc.errors()
    safe_errors = []
    for err in raw_errors:
        sanitized = _sanitize(err)
        safe_errors.append({
            "loc": sanitized.get("loc"),
            "msg": sanitized.get("msg"),
            "type": sanitized.get("type"),
            "ctx": sanitized.get("ctx"),
        })
    return JSONResponse(status_code=422, content={"detail": safe_errors})


@app.post("/extract-text", response_class=PlainTextResponse)
async def extract_text(request: Request):
    """Extract text supporting either JSON body {path}, Form path, or uploaded PDF file.

    Query parameters:
    - format: raw | plain | ai (default: plain)
      raw   -> return original OCR text (uninfluenced by cleaning/correction)
      plain -> cleaned text (whitespace/table normalized)
      ai    -> fully corrected text (current default pipeline output)
    - ocr_engine: tesseract | qwen (default: tesseract)
      tesseract -> Traditional Tesseract OCR
      qwen      -> AI-based Qwen-VL vision-language model (better for noisy documents)
    """
    # Determine requested format and OCR engine from query params
    format = request.query_params.get("format", "plain").lower()
    ocr_engine = request.query_params.get("ocr_engine", "tesseract").lower()

    content_type = request.headers.get("content-type", "")
    supplied_path: Optional[str] = None
    upload_file = None
    temp_pdf_path: Optional[str] = None
    target_pdf: Optional[Path] = None

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            # path may be included in form
            if "path" in form:
                supplied_path = _normalize_path(str(form.get("path")))
            if "file" in form:
                upload_file = form.get("file")  # starlette.datastructures.UploadFile
        else:
            body_bytes = await request.body()
            if body_bytes:
                try:
                    import json
                    data = json.loads(body_bytes.decode("utf-8", errors="ignore"))
                    supplied_path = _normalize_path(data.get("path"))
                except Exception:
                    # body may be raw PDF (incorrectly posted) -> write to temp and treat as file
                    if len(body_bytes) > 32 and body_bytes.startswith(b"%PDF"):
                        import tempfile
                        fd, temp_pdf_path = tempfile.mkstemp(prefix="rawbody_", suffix=".pdf")
                        os.close(fd)
                        with open(temp_pdf_path, "wb") as f:
                            f.write(body_bytes)
                        target_pdf = Path(temp_pdf_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse request body: {e}")

    # If we got an upload via multipart
    if upload_file is not None and target_pdf is None:
        filename = upload_file.filename or "uploaded.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
        import tempfile, shutil
        fd, temp_pdf_path = tempfile.mkstemp(prefix="uploaded_", suffix=".pdf")
        os.close(fd)
        with open(temp_pdf_path, "wb") as out:
            shutil.copyfileobj(upload_file.file, out)
        target_pdf = Path(temp_pdf_path)

    # Path based
    if target_pdf is None and supplied_path:
        target_pdf = Path(supplied_path)
        if not target_pdf.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {supplied_path}")
        if target_pdf.suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail=f"Not a PDF file: {target_pdf.suffix}")

    if target_pdf is None:
        raise HTTPException(status_code=400, detail="No PDF provided (upload 'file' or JSON {'path': ...})")

    try:
        temp_output = f"data/json/temp_{target_pdf.stem}.json"
        record = run_pipeline(
            input_pdf=str(target_pdf),
            output_json=temp_output,
            data_root="data",
            max_workers=1,
            enable_embeddings=False,  # disable heavy embedding model for extraction endpoint
            ocr_engine=ocr_engine  # Pass selected OCR engine
        )
        # Choose format
        fmt = format.lower()
        def _segments_for(fmt_key: str) -> str:
            seg_key = {
                "raw": "segments_raw",
                "plain": "segments_clean",
                "ai": "segments_corrected",
            }.get(fmt_key, "segments_corrected")
            segments = record.metadata.get(seg_key)
            if isinstance(segments, list) and segments:
                # Each announcement followed by === line
                return "\n===\n".join(s.strip() for s in segments if s.strip()) + "\n===\n"
            # Fallback to single text
            if fmt_key == "raw":
                return record.metadata.get("raw_text") or record.announcement or "(No text)"
            if fmt_key == "plain":
                return record.metadata.get("clean_text") or record.announcement or "(No text)"
            return record.announcement or "(No text)"
        return _segments_for(fmt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception:
                pass


@app.post("/extract-images")
def extract_images(req: ProcessRequest):
    """
    Extract images from PDF document.
    
    Returns list of images with base64 encoded data.
    """
    pdf = Path(req.path)
    if not pdf.exists() or pdf.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Invalid or missing PDF path")
    try:
        import fitz  # PyMuPDF
        import base64
        
        doc = fitz.open(str(pdf))
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Convert to base64
                b64 = base64.b64encode(image_bytes).decode('utf-8')
                images.append({
                    "page": page_num + 1,
                    "index": img_index + 1,
                    "format": image_ext,
                    "data": f"data:image/{image_ext};base64,{b64}"
                })
        
        doc.close()
        return {"ok": True, "count": len(images), "images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
