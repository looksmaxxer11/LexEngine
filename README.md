# DeepSeek Document Intelligence Pipeline

> **Next-generation AI-powered document processing system using DeepSeek-OCR**

A production-grade Python pipeline that converts complex multilingual (Uzbek Latin, Uzbek Cyrillic, Russian, English) newspaper-style PDFs into clean, structured Markdown documents with intelligent layout understanding, table detection, and adaptive learning capabilities.

---

## 🌟 Features

### Core Capabilities
- **🤖 100% DeepSeek-OCR Integration** - No Tesseract, no PaddleOCR, pure DeepSeek intelligence
- **🌍 Multilingual Support** - Uzbek (Latin & Cyrillic), Russian, English with script detection
- **📰 Complex Layout Understanding** - Multi-column newspapers, announcements, mixed scripts
- **📊 Intelligent Table Detection** - Automatic table extraction and Markdown formatting
- **🔧 Homoglyph Correction** - Smart Cyrillic/Latin character disambiguation
- **📈 Adaptive Learning** - Self-improvement through training data collection
- **⚡ GPU Acceleration** - CUDA support for fast processing
- **🎯 High Accuracy** - Advanced preprocessing and confidence scoring

### Advanced Features
- Script-aware OCR with language detection
- Announcement structure recognition
- Title/header detection based on font size and keywords
- Multi-column layout merging (left-to-right)
- Header/footer removal
- Quote normalization
- Training dataset generation for fine-tuning
- JSON debug output for analysis
- Batch processing mode
- Comprehensive logging and error tracking

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- DeepSeek API key ([Get one here](https://www.deepseek.com))
- Windows/Linux/macOS with 8GB+ RAM
- (Optional) NVIDIA GPU with CUDA for acceleration

### Installation

1. **Clone or download this repository**
   ```bash
   cd "AI for text processing"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure DeepSeek API**
   
   Edit `config.ini` and add your API key:
   ```ini
   [DEEPSEEK]
   api_key = YOUR_DEEPSEEK_API_KEY_HERE
   ```

4. **Create directory structure** (automatic on first run)
   ```
   input/          # Place your PDF files here
   output/         # Processed Markdown files
   logs/           # Application logs
   training_data/  # Training samples
   models/         # Fine-tuned models
   temp/           # Temporary files
   ```

### Basic Usage

**Process a single PDF:**
```bash
python cli.py process --input "input/document.pdf" --output "output/document.md" --gpu
```

**Batch process all PDFs in a directory:**
```bash
python cli.py batch --input-dir input/ --output-dir output/
```

**Export training dataset:**
```bash
python cli.py export-dataset --output dataset.jsonl
```

**Analyze training data:**
```bash
python cli.py analyze --input training_data --output report.json
```

**Fine-tune model (requires training API access):**
```bash
python cli.py train --dataset training_data --epochs 3 --gpu
```

---

## 📖 Detailed Documentation

### Command Reference

#### 1. Process Command
Convert PDF to Markdown with intelligent parsing.

```bash
python cli.py process [OPTIONS]
```

**Options:**
- `-i, --input PATH` - Input PDF file (required)
- `-o, --output PATH` - Output Markdown file (required)
- `-c, --config PATH` - Config file (default: config.ini)
- `--gpu / --no-gpu` - GPU acceleration (default: enabled)
- `--log-uncertain / --no-log-uncertain` - Log uncertain OCR outputs (default: enabled)
- `--json-output PATH` - Save JSON debug output
- `-v, --verbose` - Enable verbose logging

**Example:**
```bash
python cli.py process \
  --input "official_announcements.pdf" \
  --output "output/announcements.md" \
  --json-output "output/debug.json" \
  --gpu \
  --verbose
```

#### 2. Batch Command
Process multiple PDFs at once.

```bash
python cli.py batch [OPTIONS]
```

**Options:**
- `-i, --input-dir PATH` - Input directory (default: input)
- `-o, --output-dir PATH` - Output directory (default: output)
- `-c, --config PATH` - Config file (default: config.ini)

**Example:**
```bash
python cli.py batch -i "C:/documents/pdfs" -o "C:/documents/markdown"
```

#### 3. Train Command
Fine-tune DeepSeek model on custom data.

```bash
python cli.py train [OPTIONS]
```

**Options:**
- `-d, --dataset PATH` - Training dataset directory (required)
- `-o, --output-model PATH` - Output model directory (default: models/finetuned)
- `-e, --epochs INT` - Training epochs (default: 3)
- `-b, --batch-size INT` - Batch size (default: 8)
- `-lr, --learning-rate FLOAT` - Learning rate (default: 2e-5)
- `--gpu / --no-gpu` - GPU training (default: enabled)

**Example:**
```bash
python cli.py train \
  --dataset training_data \
  --output-model models/uzbek_press_v1 \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 1e-5 \
  --gpu
```

#### 4. Analyze Command
Analyze training data and identify common errors.

```bash
python cli.py analyze [OPTIONS]
```

**Options:**
- `-i, --input PATH` - Training data directory (required)
- `-o, --output PATH` - Output report file (JSON)

**Example:**
```bash
python cli.py analyze -i training_data -o reports/analysis_2024.json
```

#### 5. Export Dataset Command
Export training samples to JSONL format.

```bash
python cli.py export-dataset [OPTIONS]
```

**Options:**
- `-o, --output PATH` - Output JSONL file (required)
- `-d, --training-dir PATH` - Training directory (default: training_data)

**Example:**
```bash
python cli.py export-dataset -o datasets/uzbek_press_dataset.jsonl
```

---

## ⚙️ Configuration

### config.ini Structure

```ini
[DEEPSEEK]
api_key = YOUR_KEY_HERE
base_url = https://api.deepseek.com
model = deepseek-chat
temperature = 0.1
max_tokens = 4096

[OCR]
confidence_threshold = 0.70
batch_size = 4
use_gpu = true
dpi = 300
enhance_contrast = true
denoise = true
deskew = true

[LANGUAGE]
supported_languages = uz_latin,uz_cyrillic,ru,en
latin_threshold = 0.6
cyrillic_threshold = 0.6

[ANNOUNCEMENT]
# Keywords for announcement detection
uz_latin_keywords = E'LON,ELON,DIQQAT
uz_cyrillic_keywords = ЭЪЛОН,ДИҚҚАТ
russian_keywords = ОБЪЯВЛЕНИЕ,ВНИМАНИЕ
english_keywords = ANNOUNCEMENT,NOTICE,ATTENTION

[TABLE]
min_columns = 2
min_rows = 2
cell_padding = 1

[OUTPUT]
output_format = markdown
bold_titles = true
normalize_quotes = true
preserve_numbers = true

[TRAINING]
training_data_dir = training_data
low_confidence_threshold = 0.60
log_uncertain_outputs = true
learning_rate = 2e-5
num_epochs = 3

[PATHS]
input_dir = input
output_dir = output
logs_dir = logs
models_dir = models
```

### Environment Variables

Alternatively, set API key via environment variable:

**Windows (PowerShell):**
```powershell
$env:DEEPSEEK_API_KEY = "your-api-key"
```

**Linux/Mac:**
```bash
export DEEPSEEK_API_KEY="your-api-key"
```

---

## 🏗️ Architecture

### Pipeline Components

```
┌─────────────────────────────────────────────────────────┐
│                 DeepSeekDocumentPipeline                │
└─────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
│ DeepSeekOCR      │ │ ScriptDetector│ │ Announcement │
│ Processor        │ │               │ │ Parser       │
├──────────────────┤ ├──────────────┤ ├──────────────┤
│ • PDF to Image   │ │ • Latin/Cyr  │ │ • Title      │
│ • Preprocessing  │ │   Detection  │ │   Detection  │
│ • DeepSeek API   │ │ • Homoglyph  │ │ • Content    │
│ • Layout Analysis│ │   Correction │ │   Merging    │
│ • Confidence     │ │ • Language   │ │ • Structure  │
│   Scoring        │ │   ID         │ │   Parsing    │
└──────────────────┘ └──────────────┘ └──────────────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
           ┌───────────────────────────────┐
           │      TableFormatter           │
           ├───────────────────────────────┤
           │ • Table Detection             │
           │ • Cell Grouping               │
           │ • Markdown Generation         │
           └───────────────────────────────┘
                           │
                           ▼
           ┌───────────────────────────────┐
           │      MarkdownWriter           │
           ├───────────────────────────────┤
           │ • Format Announcements        │
           │ • Add Metadata                │
           │ • Write Output                │
           └───────────────────────────────┘
                           │
                           ▼
           ┌───────────────────────────────┐
           │      TrainingManager          │
           ├───────────────────────────────┤
           │ • Log Uncertain Outputs       │
           │ • Dataset Export              │
           │ • Fine-tuning Prep            │
           └───────────────────────────────┘
```

### Class Overview

#### DeepSeekOCRProcessor
- PDF to image conversion
- Image preprocessing (denoise, enhance, deskew)
- DeepSeek Vision API integration
- Text block extraction with bounding boxes
- Confidence scoring

#### ScriptDetector
- Unicode-based script detection (Latin/Cyrillic)
- Language identification (Uzbek/Russian/English)
- Homoglyph correction
- Text normalization

#### AnnouncementParser
- Keyword-based announcement detection
- Font size analysis for titles
- Text block merging
- Paragraph reconstruction

#### TableFormatter
- Cell proximity analysis
- Row/column grouping
- Markdown table generation
- Header detection

#### MarkdownWriter
- Document formatting
- Metadata injection
- Quote normalization
- File output management

#### TrainingManager
- Low-confidence sample logging
- Image + label pair creation
- JSONL dataset export
- Training data analysis

---

## 📊 Output Format

### Markdown Structure

```markdown
<!-- Document Metadata -->
<!-- Source: document.pdf -->
<!-- Pages: 10 -->
<!-- Announcements: 14 -->
<!-- Processed: 2024-11-13T10:30:00 -->

**E'LON - TENDERGA TAKLIF**

Davlat xarid qilish markazi 2024-yil 15-dekabr kuni soat 10:00 da 
ochiq tender o'tkazadi. Tender predmeti: kompyuter texnikasi va 
dasturiy ta'minot.

Ishtirok etish shartlari:
- Yuridik shaxs bo'lishi
- Soliq to'lovchi sertifikati
- Bank kafolat xati

===


**ДИҚҚАТ - ЯНГИ ҚОИДАЛАР**

2024 йилдан бошлаб қуйидаги қоидалар кучга киради...

===


**ANNOUNCEMENT - NEW REGULATIONS**

Starting from 2024, the following regulations will be in effect...

| Item       | Price  | Quantity |
| ---------- | ------ | -------- |
| Computer   | $500   | 10       |
| Monitor    | $200   | 15       |

===

```

### JSON Debug Output

```json
{
  "metadata": {
    "source": "document.pdf",
    "processed_at": "2024-11-13T10:30:00",
    "total_pages": 10,
    "total_blocks": 245,
    "processing_time": 34.5
  },
  "statistics": {
    "avg_confidence": 0.89,
    "script_distribution": {
      "latin": 120,
      "cyrillic": 95,
      "mixed": 30
    },
    "language_distribution": {
      "uz": 150,
      "ru": 70,
      "en": 25
    }
  },
  "announcements": [
    {
      "title": "E'LON - TENDERGA TAKLIF",
      "content": ["Full announcement text..."],
      "tables": [],
      "metadata": {"block_count": 15},
      "confidence": 0.92
    }
  ]
}
```

---

## 🎯 Use Cases

### 1. Government Announcements
Process official gazettes and announcement bulletins in multiple languages.

### 2. Newspaper Digitization
Convert scanned newspaper pages with complex multi-column layouts.

### 3. Legal Documents
Extract structured information from mixed-script legal documents.

### 4. Academic Papers
Process multilingual research papers with tables and figures.

### 5. Historical Archives
Digitize historical documents with Cyrillic and Latin scripts.

---

## 🔧 Troubleshooting

### Issue: "Import errors" when running
**Solution:** Install all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "DeepSeek API key not found"
**Solution:** Set your API key in `config.ini` or environment variable:
```powershell
$env:DEEPSEEK_API_KEY = "your-key"
```

### Issue: Low OCR accuracy
**Solutions:**
1. Increase DPI in config.ini: `dpi = 400`
2. Enable preprocessing: `enhance_contrast = true`
3. Fine-tune model on your specific documents
4. Check if PDF is too low quality

### Issue: GPU not being used
**Solution:** 
1. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Install GPU-enabled PyTorch: Visit [pytorch.org](https://pytorch.org)

### Issue: Tables not detected
**Solutions:**
1. Lower `min_columns` and `min_rows` in config.ini
2. Improve scan quality
3. Check if table structure is clear

### Issue: Wrong script detection
**Solution:** Adjust thresholds in config.ini:
```ini
latin_threshold = 0.7
cyrillic_threshold = 0.7
```

---

## 🚀 Performance Tips

### For Speed
- Use GPU acceleration (`--gpu`)
- Lower DPI for draft processing: `dpi = 200`
- Disable preprocessing for clean PDFs
- Use batch mode for multiple files

### For Accuracy
- Increase DPI: `dpi = 400`
- Enable all preprocessing options
- Fine-tune model on your domain
- Collect and label uncertain outputs
- Adjust confidence thresholds

### For Large Documents
- Process in batches
- Monitor memory usage
- Use temp directory cleanup
- Enable rotation in logs

---

## 📚 Advanced Topics

### Fine-Tuning Workflow

1. **Process documents and collect uncertain outputs:**
   ```bash
   python cli.py process -i doc.pdf -o out.md --log-uncertain
   ```

2. **Analyze training data:**
   ```bash
   python cli.py analyze -i training_data -o report.json
   ```

3. **Export dataset:**
   ```bash
   python cli.py export-dataset -o dataset.jsonl
   ```

4. **Fine-tune model:**
   ```bash
   python cli.py train -d training_data -e 5 --gpu
   ```

5. **Update config to use fine-tuned model**

### Custom Script Detection

Edit `document_processor.py` and modify `ScriptDetector` class:

```python
class ScriptDetector:
    # Add custom character ranges
    CUSTOM_CHARS = r'[your_chars_here]'
    
    # Add custom word dictionaries
    CUSTOM_WORDS = {'word1', 'word2'}
```

### Adding New Languages

1. Add to config.ini:
   ```ini
   [LANGUAGE]
   supported_languages = uz_latin,uz_cyrillic,ru,en,new_lang
   ```

2. Add keywords:
   ```ini
   [ANNOUNCEMENT]
   new_lang_keywords = KEYWORD1,KEYWORD2
   ```

3. Update `ScriptDetector` class with new word lists

---

## 🤝 Contributing

This is a production-ready codebase. To contribute:

1. Fork the repository
2. Create feature branch
3. Follow PEP 8 style guidelines
4. Add type hints and docstrings
5. Test thoroughly
6. Submit pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **DeepSeek** for the powerful OCR and vision models
- **OpenAI** for the API client library
- **PyTorch** for deep learning infrastructure
- **pdf2image** for PDF conversion
- **Click** for beautiful CLI

---

## 📞 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check the troubleshooting section
- Review logs in `logs/` directory

---

## 🎓 Citation

If you use this system in research or production:

```bibtex
@software{deepseek_document_intelligence,
  title = {DeepSeek Document Intelligence Pipeline},
  author = {AI Engineering Team},
  year = {2024},
  url = {https://github.com/yourusername/deepseek-document-intelligence}
}
```

---

**Built with ❤️ using DeepSeek AI**

*Version 1.0.0 | Last Updated: November 2024*
#   L e x E n g i n e  
 