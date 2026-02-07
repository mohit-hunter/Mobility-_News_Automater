# mobility News Auto-mater

> **⚠️ BETA VERSION - Work in Progress**  
> This project is currently in active development. Many features are experimental and subject to change. Contributions and feedback are welcome!

## 📋 Overview

Frost News Collector is an intelligent RSS feed aggregator and news classification system designed for collecting, deduplicating, and categorizing mobility industry news. The system features a FastAPI backend with a web-based frontend for managing news feeds and reviewing classified articles.

### Current Status: Beta Phase

This is **not a final version**. The following areas are actively being developed and improved:

- ✅ **Implemented**: RSS feed scraping, deduplication, basic classification
- 🚧 **In Progress**: Human-in-the-loop corrections, authentication system
- 📝 **Planned**: Advanced ML models, real-time updates, analytics dashboard
- 🔮 **Future**: Multi-language support, sentiment analysis, trend detection

---

## 🚀 Features

### Core Functionality
- **RSS Feed Management**: Add, remove, and monitor multiple RSS news sources
- **Intelligent Deduplication**: Fuzzy matching algorithm to identify duplicate articles across sources
- **Executive Summaries**: Automatically generates concise, executive-friendly headlines and summaries
- **Date-Range Filtering**: Query news articles by date range and search terms
- **JSON Archive**: Persistent storage of all collected news in `master_news_archive.json`

### Beta Features (Experimental)
- **AI Classification**: Multinomial logistic regression classifier for categorizing news into:
  - `automakers` - News about automotive manufacturers
  - `government` - Policy, regulation, and government initiatives
  - `suppliers` - Supply chain and component suppliers
- **Authentication System**: Simple JSON-based user management (beta users only)
- **Human-in-the-Loop**: Reviewers can correct AI predictions to improve accuracy
- **Role-Based Access**: Admin and Reviewer roles for quality control

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd Frost_News_Collector
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure the application**
   - Edit `config.py` to enable/disable beta features
   - Set `BETA_MODE = False` to disable experimental features

6. **Run the application**
   ```bash
   python main.py
   ```

7. **Access the web interface**
   - Open your browser to: `http://127.0.0.1:8000`
   - API documentation: `http://127.0.0.1:8000/docs`

---

## 📁 Project Structure

```
Frost_News_Collector/
├── main.py                          # FastAPI application entry point
├── services.py                      # Core scraping and data processing logic
├── rss.py                           # RSS feed parsing utilities
├── config.py                        # Configuration and beta mode toggle
├── beta_auth.py                     # Beta authentication system
├── beta_classifier.py               # AI classification service
├── feeds.json                       # RSS feed sources configuration
├── master_news_archive.json         # Persistent news archive
├── beta_users.json                  # Beta user accounts (beta mode)
├── beta_label_corrections.json      # Human corrections (beta mode)
├── Classifier/
│   ├── baseline_classifier.py       # Classifier training script
│   ├── baseline_model.pkl           # Trained model
│   ├── News_Semantic_Embedding_v1.npy  # Precomputed embeddings
│   ├── embedding_index.json         # Embedding lookup index
│   └── training_metrics.json        # Model performance metrics
├── Embeddings/                      # Embedding generation scripts
├── static/                          # Frontend HTML/CSS/JS files
└── requirements.txt                 # Python dependencies
```

---

## 🔧 Configuration

### Beta Mode Toggle

Edit `config.py` to control beta features:

```python
# Set to False to disable all beta features
BETA_MODE = True  # or False

# Classification categories
CATEGORIES = ["automakers", "government", "suppliers"]
```

When `BETA_MODE = False`:
- Authentication endpoints are disabled
- Classification is skipped
- Original scraping functionality remains intact

---

## 📡 API Endpoints

### Core Endpoints (Always Available)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/news` | Retrieve news with optional filters (`start_date`, `end_date`, `search`) |
| `POST` | `/api/scrape` | Manually trigger the news scraper |
| `GET` | `/api/feeds` | List all configured RSS feeds |
| `POST` | `/api/feeds` | Add a new RSS feed |
| `DELETE` | `/api/feeds/{name}` | Remove an RSS feed |

### Beta Endpoints (Only if `BETA_MODE = True`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/login` | Login with email (no password in beta) |
| `POST` | `/api/auth/logout` | Invalidate authentication token |
| `GET` | `/api/auth/me` | Get current user information |
| `POST` | `/api/corrections` | Submit a label correction (reviewers only) |
| `GET` | `/api/categories` | Get available classification categories |

---

## 🧪 Machine Learning Pipeline

### Current Implementation (Baseline)

The classifier uses a **multinomial logistic regression** model trained on semantic embeddings:

1. **Embeddings**: Precomputed sentence embeddings stored in `News_Semantic_Embedding_v1.npy`
2. **Model**: Trained baseline classifier in `baseline_model.pkl`
3. **Inference**: Real-time classification using embedding lookup
4. **Corrections**: Human feedback stored in `beta_label_corrections.json`

### Training the Classifier

To retrain the model with new data:

```bash
cd Classifier
python baseline_classifier.py
```

This will:
- Load training data from `news.csv`
- Generate semantic embeddings
- Train a logistic regression model
- Save model artifacts and metrics

### Known Limitations (To Be Improved)

- ⚠️ **Limited training data**: Current model trained on small dataset
- ⚠️ **No active learning**: Corrections are logged but not automatically retrained
- ⚠️ **Simple model**: Logistic regression may not capture complex patterns
- ⚠️ **No confidence thresholding**: All predictions returned regardless of confidence

---

## 🔐 Authentication (Beta)

### Beta User Management

Users are stored in `beta_users.json`:

```json
{
  "users": [
    {
      "user_id": "user_001",
      "email": "admin@example.com",
      "role": "admin",
      "is_active": true
    }
  ]
}
```

### Roles

- **Admin**: Full access, can review and correct classifications
- **Reviewer**: Can review and correct classifications
- **Viewer**: Read-only access (planned)

### Login Flow

1. POST email to `/api/auth/login`
2. Receive JWT-like token
3. Include token in `Authorization: Bearer <token>` header
4. Token stored in `beta_users.json` (simple file-based session)

**⚠️ Security Warning**: This is a **beta authentication system** for development only. Do NOT use in production without implementing proper security measures (password hashing, secure token storage, HTTPS, etc.).

---

## 📊 Data Flow

```
RSS Feeds → Scraper → Deduplication → Executive Summary → JSON Archive
                                              ↓
                                      (if BETA_MODE)
                                              ↓
                                    Semantic Embedding Lookup
                                              ↓
                                      Classifier Prediction
                                              ↓
                                    Human Correction (optional)
                                              ↓
                                         API Response
```

---

## 🚧 Known Issues & Limitations

### Current Bugs
- [ ] Embedding index may be out of sync with master archive for new articles
- [ ] No pagination for large result sets
- [ ] Token expiration not enforced
- [ ] Duplicate cluster IDs across scraping runs

### Performance Issues
- [ ] Deduplication is O(n²) - slow for large datasets
- [ ] No caching for classification results
- [ ] Frontend loads entire news archive (no lazy loading)

### Missing Features
- [ ] Scheduled scraping (currently manual trigger only)
- [ ] Email notifications for new articles
- [ ] Export to CSV/Excel
- [ ] Analytics dashboard
- [ ] Multi-user collaboration features
- [ ] API rate limiting
- [ ] Comprehensive error logging

---

## 🛣️ Roadmap

### Short-Term (Next Release)
- [ ] Implement automatic retraining with human corrections
- [ ] Add confidence threshold filtering
- [ ] Improve frontend UX/UI
- [ ] Add pagination for API responses
- [ ] Implement proper logging system

### Mid-Term
- [ ] Upgrade to transformer-based classification (BERT, RoBERTa)
- [ ] Add scheduled scraping with cron/celery
- [ ] Implement real-time WebSocket updates
- [ ] Create analytics dashboard
- [ ] Add export functionality

### Long-Term
- [ ] Multi-language support
- [ ] Sentiment analysis
- [ ] Trend detection and forecasting
- [ ] Integration with external APIs (Twitter, LinkedIn)
- [ ] Mobile app

---

## 🤝 Contributing

This project is in active development and welcomes contributions!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Areas Needing Help
- Improving classification accuracy
- Frontend development (React/Vue migration?)
- Performance optimization
- Documentation
- Testing and QA

---

## 📝 License

This项目 is currently unlicensed. Please contact the project maintainer for usage permissions.

---

## 📧 Contact & Support

For questions, bug reports, or feature requests, please open an issue on the project repository.

---

## ⚡ Quick Start Guide

**Minimal setup to get running:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python main.py

# 3. Open browser
# Navigate to http://127.0.0.1:8000

# 4. Add RSS feeds via the UI or API
# POST to /api/feeds with {"name": "TechCrunch", "url": "https://..."}

# 5. Trigger scraping
# POST to /api/scrape

# 6. View collected news
# GET /api/news
```

---

## 🔍 Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'fastapi'`  
**Solution**: Ensure you've activated the virtual environment and run `pip install -r requirements.txt`

**Problem**: Classifier not working  
**Solution**: Check that `BETA_MODE = True` in `config.py` and that model files exist in `Classifier/`

**Problem**: No news articles appearing  
**Solution**: Ensure RSS feeds are configured in `feeds.json` and run `/api/scrape` to collect articles

**Problem**: Authentication not working  
**Solution**: Verify `BETA_MODE = True` and check `beta_users.json` contains valid user entries

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [RSS Feed Specification](https://www.rssboard.org/rss-specification)
- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

**Remember**: This is a **beta version** under active development. Expect breaking changes, bugs, and incomplete features. Your feedback and contributions are invaluable in making this project better! 🚀
