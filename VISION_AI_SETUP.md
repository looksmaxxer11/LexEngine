# Vision AI Setup Guide

## What is Vision AI?

Vision AI mode uses **GPT-4 with Vision** (or similar multimodal models) to actually **SEE and READ** your scanned documents. Unlike traditional OCR which analyzes pixels, Vision AI:

- **Understands** document layouts and context
- **Reads** mixed scripts (Uzbek Latin/Cyrillic, Russian, English)
- **Handles** handwriting, stamps, and complex formatting
- **Extracts** text with high accuracy

## Quick Setup

### 1. Install OpenAI Package

```bash
pip install openai
```

Or update your full requirements:

```bash
pip install -r requirements_pipeline.txt
```

### 2. Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy the key (starts with `sk-...`)

### 3. Set Environment Variable

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

**Windows CMD:**
```cmd
set OPENAI_API_KEY=sk-your-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### 4. Verify Setup

```python
python -c "from src.vision_ocr import test_vision_availability; print(test_vision_availability())"
```

Should show: `{'openai_installed': True, 'api_key_set': True, 'ready': True, 'message': 'Vision OCR ready'}`

## Using Vision AI

### Web UI

1. Start the server: `start_server.bat` or `uvicorn src.server:app --reload`
2. Open http://127.0.0.1:8000
3. Upload your PDF
4. ✅ **Check "🔬 Use Vision AI"** 
5. ⚠️ **Uncheck "Dry run"** (vision needs actual images)
6. Click "Process"

### Command Line

```bash
python -m src.orchestrator \
  --input data/raw_pdfs/announcement.pdf \
  --output data/json/announcement.json \
  --use-vision \
  --data-root data
```

## Cost Estimate

GPT-4o with Vision pricing (as of Nov 2024):
- **Input**: ~$2.50 per 1M tokens (~$0.01 per image)
- **Output**: ~$10.00 per 1M tokens

For a typical 1-2 page announcement:
- **Cost per document**: ~$0.02 - $0.05
- **100 documents**: ~$2 - $5

Much cheaper than manual processing and more accurate than basic OCR!

## Troubleshooting

### "OpenAI API key required"
- Set `OPENAI_API_KEY` environment variable
- Restart terminal/server after setting

### "Rate limit exceeded"
- Free tier: 3 requests/min, 200 requests/day
- Paid tier: 10,000 requests/min
- Add delays between batch processing

### "Module 'openai' not found"
```bash
pip install openai
```

### Vision returns empty/wrong text
- Check image quality (should be 300 DPI)
- Verify PDF preprocessing worked
- Try adjusting the prompt in `src/vision_ocr.py`

## Alternative: Azure OpenAI

If you have Azure OpenAI access:

1. Set these environment variables:
```powershell
$env:AZURE_OPENAI_KEY="your-azure-key"
$env:AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

2. Modify `src/vision_ocr.py` to use Azure client:
```python
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
)
```

## Next Steps

Once Vision AI is working:
1. Process your announcement PDFs
2. Verify the extracted text quality
3. Optionally combine with post-correction (`TextCorrector`) for even better results
4. Store results in Qdrant/Postgres for search and retrieval
