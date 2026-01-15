# Fake News Detection & Summarization

Advanced ML system using **RNN + GNN** for fake news classification with **Pegasus summarization**. This project combines deep learning techniques to detect fake news and provide summaries of verified real news articles.

---

## Features

| Feature | Description |
|---------|-------------|
| **RNN (LSTM)** | Sequential text classification for news analysis |
| **GNN (GCN)** | Graph-based relational learning |
| **Pegasus** | Abstractive summarization for real news |
| **Ensemble Model** | Combined prediction logic for higher accuracy |
| **Interactive Interface** | Live news input with instant prediction |
| **Dual Datasets** | Fast testing and full accuracy modes |

---

## Datasets

The project includes multiple datasets for different use cases:

- **Fake_trim.csv / True_trim.csv** - Trimmed datasets for fast testing and development
- **Fake.csv / True.csv** - Full datasets for maximum accuracy

All datasets are included in the repository.

---

## Quick Start

### Prerequisites

- Python 3.7+
- pip package manager
- CUDA (optional, for GPU acceleration)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Jinam-Shah/Fake-News-Detection-and-Summarization.git
cd fake-news-detection
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the application**

```bash
python fake_news_detection.py
```

### Configuration

Modify datasets in code (line ~45) for different accuracy levels:

```python
# Fast testing (trimmed dataset)
fake_news_df = pd.read_csv('Fake_trim.csv')
real_news_df = pd.read_csv('True_trim.csv')

# Full accuracy (complete dataset)
fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')
```

---

## Model Performance

| Model | Test Accuracy |
|-------|--------------|
| **RNN (LSTM)** | 92% |
| **GNN (GCN)** | 91% |
| **Ensemble** | 93%+ |

---

## 🛠 Tech Stack

```
Deep Learning: PyTorch, torch-geometric
NLP: Transformers (Hugging Face), NLTK
ML Libraries: scikit-learn, pandas, numpy
Summarization: Pegasus (Google)
Graph Processing: PyTorch Geometric
```

### Key Technologies

- **PyTorch** - Deep learning framework
- **torch-geometric** - Graph neural network implementation
- **Transformers** - Pre-trained models (Pegasus)
- **NLTK** - Natural language processing
- **scikit-learn** - Machine learning utilities

---

## Research Documentation

This project is backed by comprehensive research documentation:

- **RESEARCH_PAPER.md** - Complete methodology and implementation details
- **research_paper.pdf** - Original research paper with theoretical background

Both documents are included in the repository for reference.

---

## How It Works

### Workflow

1. **Input**: User enters news article text
2. **Processing**: Text is preprocessed and tokenized
3. **RNN Analysis**: LSTM model analyzes sequential patterns
4. **GNN Analysis**: Graph Convolutional Network examines relational features
5. **Ensemble Decision**: Combined prediction from both models
6. **Output**: Classification result (True/Fake)
7. **Summarization**: If news is real, Pegasus generates a concise summary

### Live Demo Usage

```bash
# Run the application
python fake_news_detection.py

# Enter your news text when prompted
# Receive instant classification + summary (for real news)
```

**Example Output:**
```
Enter news text → "Climate change report reveals..." 
Result: TRUE
Summary: Climate report shows rising temperatures...
```

---

## Project Structure

```
fake-news-detection/
├── fake_news_detection.py    # Main implementation
├── Fake_trim.csv             # Trimmed fake news dataset
├── True_trim.csv             # Trimmed real news dataset
├── Fake.csv                  # Full fake news dataset
├── True.csv                  # Full real news dataset
├── requirements.txt          # Python dependencies
├── RESEARCH_PAPER.md         # Research methodology
├── research_paper.pdf        # Original research paper
└── README.md                 # This file
```

---

## Development

### Requirements File

Create `requirements.txt` with:

```txt
torch>=1.9.0
torch-geometric>=2.0.0
transformers>=4.0.0
nltk>=3.6
scikit-learn>=0.24.0
pandas>=1.3.0
numpy>=1.21.0
```

### Training Your Own Model

1. Prepare your dataset in CSV format with `text` and `label` columns
2. Modify dataset paths in `fake_news_detection.py`
3. Adjust hyperparameters as needed
4. Run training: `python fake_news_detection.py --train`

### Model Architecture

- **RNN Layer**: LSTM with 128 hidden units
- **GNN Layer**: 2-layer Graph Convolutional Network
- **Embedding**: 300-dimensional word embeddings
- **Optimizer**: Adam with learning rate 0.001

---

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation
- Ensure models maintain >90% accuracy

---

## Author

**Jinam Shah**

- GitHub: [@Jinam-Shah](https://github.com/Jinam-Shah)

**Jeneil Kapadia**

- GitHub: [@Jeneil-Kapadia](https://github.com/JeneilKapadia)

---
---

**Made with ❤️ for combating misinformation**
