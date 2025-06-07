# Job Market Analysis System

## 🚀 Overview
A hybrid job market analysis system that combines LLM-powered skill detection for individual job descriptions with fast, hardcoded analysis for large datasets. Built with FastAPI, Groq LLM API, and comprehensive data visualization capabilities.

## 🏗️ Architecture
- **LLM-based Analysis**: Uses Groq's Llama3-70B for intelligent skill extraction and trend classification
- **Hardcoded Analysis**: Fast regex-based skill extraction for dataset-wide analysis
- **RESTful API**: FastAPI endpoints for both single job analysis and dataset insights
- **Visualization Engine**: Matplotlib/Seaborn for comprehensive data visualizations

## 📊 Key Features
### Part 1: Data Analysis ✅
- **Skill Comparison**: Entry-level vs Senior role skill requirements
- **Top Skills Identification**: Most in-demand skills across all positions
- **Pattern Discovery**: Geographic distribution, company analysis, job title categorization

### Part 2: Data Visualization ✅
- Skills comparison charts across seniority levels
- Geographic distribution of ML/AI jobs
- Company type analysis (AI companies vs Tech giants)
- Job title category breakdown
- Top skills pie charts and bar graphs

### Part 3: Skill Trend Detector ✅
- **Smart Extraction**: LLM-powered skill identification from job descriptions
- **Trend Classification**: Categorizes skills as "emerging" or "established"
- **Confidence Scoring**: Provides trend scores (0.5-1.0) for each skill
- **Context-Aware**: Uses advanced prompting for accurate skill categorization

### Part 4: Deployment ✅
- **REST API**: FastAPI with comprehensive endpoints
- **Health Monitoring**: System health and LLM connectivity status
- **Error Handling**: Robust validation and error responses
- **Documentation**: Auto-generated API docs via Swagger UI

## 🛠️ Tech Stack
- **Backend**: Python, FastAPI, Uvicorn
- **LLM Integration**: Groq API
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Python-dotenv, OS environment variables

## 📋 Prerequisites
- Python 3.8+
- Groq API Key

## 🔧 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Arulakash-J/job-market-analysis.git
cd job-market-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Running the Application

### Start the API Server
```bash
python main.py
```

The server will start on `http://localhost:8000`

### Access API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📡 API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Skill Detection (Single Job)
```bash
POST /detect-skills
Content-Type: application/json

{
    "job_description": "Experience with PyTorch, TensorFlow, and diffusion models required..."
}
```

**Response:**
```json
{
    "detected_skills": [
        {"skill": "PyTorch", "category": "established", "trend_score": 0.82},
        {"skill": "TensorFlow", "category": "established", "trend_score": 0.91},
        {"skill": "diffusion models", "category": "emerging", "trend_score": 0.67}
    ],
    "total_skills": 3
}
```

### 3. Dataset Analysis
```bash
POST /analyze-dataset
Content-Type: application/json

{
    "csv_path": "dataset.csv"
}
```

**Response:** Comprehensive analysis including:
- Skill differences between entry/senior roles
- Top 3 in-demand skills
- Geographic patterns
- Company analysis
- Enhanced visualizations

### 4. LLM Testing
```bash
POST /test-llm
```

## 📈 Analysis Results

### Skill Differences (Entry vs Senior)
The system identifies distinct skill patterns:
- **Entry Level**: Focus on foundational skills (Python, SQL, basic ML)
- **Senior Level**: Advanced skills (MLOps, cloud platforms, architecture)

### Top In-Demand Skills
1. **Python** - Universal requirement across all levels
2. **Machine Learning** - Core competency
3. **SQL** - Data manipulation essential

### Interesting Patterns Discovered
- **Geographic Clustering**: Major tech hubs dominate job postings
- **Company Types**: AI-focused startups vs established tech giants
- **Title Evolution**: Emerging roles like "ML Engineer" vs traditional "Data Scientist"

## 🎯 Performance Optimizations
- **Hybrid Approach**: LLM for accuracy, regex for speed
- **Batch Processing**: Efficient dataset analysis
- **Caching**: In-memory skill pattern storage
- **Error Recovery**: Graceful fallbacks for LLM failures

## 🔍 Validation & Testing
- **Input Validation**: Pydantic models for request/response
- **Error Handling**: Comprehensive HTTP error responses
- **LLM Testing**: Built-in functionality verification
- **Data Quality**: Automated data cleaning and validation

## 📊 Visualization Outputs
The system generates `enhanced_job_market_analysis.png` containing:
1. Skills comparison (Entry vs Senior)
2. Geographic distribution
3. Top skills pie chart
4. Seniority level distribution
5. Company type analysis
6. Job title categories

## 🚨 Error Handling
- **Missing API Key**: Clear error messages for setup issues
- **Invalid Input**: Validation errors with specific guidance
- **LLM Failures**: Fallback mechanisms for service interruptions
- **Data Issues**: Robust handling of missing/malformed data

## 🔒 Security Features
- **API Key Protection**: Environment variable storage
- **Input Sanitization**: Prevents injection attacks
- **Rate Limiting**: Implicit through LLM API limits
- **Error Masking**: No sensitive information in error responses

## 📚 Additional Features
- **Real-time Health Monitoring**: System status endpoints
- **Comprehensive Logging**: Detailed operation tracking
- **Extensible Architecture**: Easy to add new analysis methods
- **Performance Metrics**: Processing time and success rates

## 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support
For issues or questions:
1. Check the API documentation at `/docs`
2. Review error logs for troubleshooting
3. Verify Groq API key configuration
4. Ensure dataset format compliance

## 🚀 Future Enhancements
- Real-time job posting ingestion
- Advanced NLP models for skill extraction
- Interactive web dashboard
- Historical trend analysis
- Salary prediction models