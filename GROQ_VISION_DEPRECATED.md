# 🚨 IMPORTANT: Groq Vision Models Deprecated

**Date:** November 13, 2025

## What Happened?

Groq has **deprecated all Llama 3.2 Vision models**:
- ❌ `llama-3.2-90b-vision-preview` (decommissioned)
- ❌ `llama-3.2-11b-vision-preview` (decommissioned)

Error when using these models:
```
Error code: 400 - The model `llama-3.2-90b-vision-preview` has been decommissioned 
and is no longer supported.
```

## ✅ Solution: Use OpenAI GPT-4o Vision

OpenAI's GPT-4o Vision is now the recommended provider for Vision OCR.

### Quick Setup

1. **Get OpenAI API Key** (paid, but affordable):
   - Visit: https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Copy the key (starts with `sk-proj-...`)

2. **Add to .env.local**:
   ```bash
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   ```

3. **Restart Server**:
   ```bash
   .\start_server.bat
   ```

4. **Use Vision AI**:
   - ✅ Check "🔬 Use Vision AI"
   - ✅ Check "👁️ Transcribe only" (to test)
   - ❌ Uncheck "Dry run"
   - Upload PDF and process

### Pricing (OpenAI GPT-4o Vision)

Very affordable for OCR use:
- **Input**: $2.50 per 1M tokens (~$0.0025 per 1K tokens)
- **Output**: $10.00 per 1M tokens (~$0.01 per 1K tokens)
- **Typical 3-page PDF**: ~$0.02-0.05 (2-5 cents)

Example:
- 100 PDFs (3 pages each) ≈ $2-5
- 1,000 PDFs ≈ $20-50

## Alternative: Use PaddleOCR (Free, No API)

If you don't want to use paid Vision AI:

1. **Uncheck "Use Vision AI"** in the UI
2. **Uncheck "Dry run"**
3. Process → Uses free PaddleOCR (runs locally, no API needed)

PaddleOCR is good for:
- Clean printed text
- Latin and Cyrillic scripts
- When cost is a concern

Vision AI (GPT-4o) is better for:
- Handwritten text
- Complex layouts
- Mixed scripts
- Low-quality scans
- Tables and forms

## Updated Documentation

All guides have been updated:
- ✅ `GROQ_QUICKSTART.md` → deprecated notice added
- ✅ `VISION_AI_SETUP.md` → updated for OpenAI-first workflow
- ✅ `src/vision_ocr.py` → graceful deprecation handling

## Status Check

Visit **Diagnostics** page in UI:
- Shows current provider status
- Confirms OpenAI key is set
- Tests API connectivity

## Questions?

**Q: Why did Groq deprecate these models?**
A: Groq focuses on fast inference for text models. Vision models were preview-only and have been sunset.

**Q: Will Groq add new vision models?**
A: Check https://console.groq.com/docs/models for latest updates.

**Q: Is OpenAI the only alternative?**
A: You can also use:
  - Azure OpenAI (enterprise)
  - Anthropic Claude 3 Vision (add support if needed)
  - Local models via Ollama (slower, but free)

**Q: Can I still use Groq for anything?**
A: Yes! Groq is excellent for fast text generation (non-vision tasks). But for Vision OCR, OpenAI GPT-4o is now the way to go.

---

**Bottom line**: Set `OPENAI_API_KEY` in `.env.local` and you're back in business! 🚀
