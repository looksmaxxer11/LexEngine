# 🚀 Quick Start: Vision AI with Groq (FREE!)

## ✅ Yes, Groq is FREE and has Vision!

Groq provides **Llama 3.2 90B Vision** model that can:
- 👁️ **Actually SEE and READ** your scanned documents
- ⚡ **10-20x faster** than GPT-4 Vision
- 💰 **Completely FREE** (no credit card needed!)
- 🌍 **Reads multiple scripts**: Uzbek (Latin/Cyrillic), Russian, English

## Setup (2 minutes)

### 1. Install Groq

```powershell
.\setup_vision.bat
```

Or manually:
```powershell
pip install groq
```

### 2. Get FREE API Key

1. Visit: **https://console.groq.com**
2. Sign up (no credit card!)
3. Click "API Keys" → Create new key
4. Copy the key (starts with `gsk_...`)

### 3. Set the Key

**PowerShell:**
```powershell
$env:GROQ_API_KEY="gsk-your-key-here"
```

**CMD:**
```cmd
set GROQ_API_KEY=gsk-your-key-here
```

## Test It Now!

### Option 1: Web UI (Easiest)

1. **Start the server:**
   ```powershell
   .\start_server.bat
   ```

2. **Open:** http://127.0.0.1:8000

3. **Upload your PDF announcement**

4. **Check these boxes:**
   - ✅ "🔬 Use Vision AI" 
   - ❌ "Dry run" (uncheck this!)

5. **Click "Process"**

6. **See the magic!** The AI will read your document and extract:
   - Reference numbers
   - Dates
   - Full announcement text
   - Language detection

### Option 2: Command Line

```powershell
python -m src.orchestrator `
  --input "data/raw_pdfs/announcement.pdf" `
  --output "data/json/output.json" `
  --use-vision
```

## What You'll Get

**Before (Dry Run):**
```json
{
  "announcement": "[DRY-RUN] OCR text placeholder"
}
```

**After (Vision AI):**
```json
{
  "ref": "12345",
  "date": "2022-02-03",
  "language": "uz-cyrillic",
  "announcement": "ҲУКМ\n№ 12345\n2022 йил 3 февраль\n\nТошкент шаҳар суд қарори...",
  "metadata": {
    "processed_at": "2025-11-13T14:00:00Z",
    "pages": 2
  }
}
```

## Why Groq?

| Feature | Groq | OpenAI GPT-4V |
|---------|------|---------------|
| **Cost** | FREE! | ~$0.02-0.05 per doc |
| **Speed** | ⚡ Lightning fast | Slower |
| **Quality** | Excellent | Excellent |
| **Setup** | No credit card | Requires payment |
| **Rate Limits** | 30 req/min free tier | 10 req/min paid |

## Troubleshooting

### "Module groq not found"
```powershell
pip install groq
```

### "Groq API key required"
```powershell
$env:GROQ_API_KEY="gsk-your-key-here"
```
Then restart your terminal/server.

### "Rate limit exceeded"
Free tier: 30 requests/minute, 14,400/day
(That's plenty for testing and small production use!)

### Vision returns empty text
- Make sure "Dry run" is **unchecked**
- Check that PDF preprocessing worked
- Verify the API key is set correctly

## Compare: Traditional OCR vs Vision AI

**Traditional OCR (PaddleOCR):**
- ❌ Requires Poppler, OpenCV setup
- ❌ Struggles with mixed scripts
- ❌ Manual layout detection
- ⚠️ Medium accuracy

**Vision AI (Groq):**
- ✅ Zero setup (just API key)
- ✅ Understands context
- ✅ Reads any script
- ✅ High accuracy
- ✅ FREE!

## Next Steps

1. ✅ Set your `GROQ_API_KEY`
2. ✅ Process your first announcement
3. ✅ See it actually read your documents!

## Full Documentation

- **Vision Setup**: `VISION_AI_SETUP.md`
- **API Reference**: `API_REFERENCE.md` (if exists)
- **Pipeline Overview**: `PROJECT_SUMMARY.md`

---

**🎉 You now have FREE AI vision that can actually read your legal announcements!**
