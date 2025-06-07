import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import re
from collections import Counter
import os
from groq import Groq
from dotenv import load_dotenv
import warnings
import uvicorn
from typing import List, Dict, Optional
import json

warnings.filterwarnings('ignore')

load_dotenv()

class JDInput(BaseModel):
    job_description: str

class SkillData(BaseModel):
    skill: str
    category: str
    trend_score: float

class SkillResponse(BaseModel):
    detected_skills: List[SkillData]
    total_skills: int

class DatasetRequest(BaseModel):
    csv_path: str = "dataset.csv"

class StatusResponse(BaseModel):
    status: str
    message: str

class TextParser:
    # skill extraction using LLM
    
    def __init__(self, groq_client):
        self.client = groq_client
        
    def get_skills_from_text(self, job_desc):
        # simple prompt to get skills list
        try:
            prompt = (f"can you just list all the actual tech skills and tools mentioned in this job desc?\n"
    f"{job_desc}\n\n"
    "just list things like languages, frameworks, databases, platforms, etc. not soft stuff like 'experience' or 'familiar with'\n"
    "Do Not use intro lines like 'Here is the list of tech skills and tools' — just a comma-separated list of concrete tech terms. avoid repeats."
)
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192", 
                max_tokens=300,
                temperature=0.1
            )
            
            skills_text = response.choices[0].message.content.strip()
            skills = [s.strip() for s in skills_text.split(',')]
            return [s for s in skills if s and len(s.strip()) > 1]
            
        except Exception as e:
            print(f"skill extraction failed: {e}")
            return []
    
    def classify_skill(self, skill):
        try:
            prompt = f"""
hey, quick check — for the skill below, would you say it's more of an "emerging" tech or something already "established"?

skill: {skill}

basically:
- call it "emerging" if it's a newer tool, library, or trend (like stuff that's caught on in the last couple years)
- call it "established" if it's been widely used or mainstream for a while

a few examples to help:
- Python → established  
- React → established  
- ChatGPT/GPT → emerging  
- Kubernetes → established  
- LangChain → emerging  
- TensorFlow → established  
- Diffusion Models → emerging  
- SQL → established

also — just give me a confidence score between 0.5 and 1.0 on how sure you are.

format it like this:
Category: [emerging or established]  
Score: [number between 0.5 and 1.0]
"""

            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                max_tokens=100,
                temperature=0.2
            )
            
            result = response.choices[0].message.content.strip()
            
            # parse response
            category = "established"  
            score = 0.7  
            
            for line in result.split('\n'):
                if 'Category:' in line:
                    category = line.split('Category:')[1].strip().lower()
                elif 'Score:' in line:
                    try:
                        score = float(line.split('Score:')[1].strip())
                    except:
                        score = 0.7
            
            return category, min(max(score, 0.5), 1.0)
            
        except Exception as e:
            print(f"classification failed for {skill}: {e}")
            # fallback logic - rough heuristics
            if any(term in skill.lower() for term in ['gpt', 'llm', 'diffusion', 'langchain', 'chatgpt']):
                return "emerging", 0.6
            else:
                return "established", 0.7

class DataAnalyzer:
    # fast analyzer for processing datasets
    
    def __init__(self, csv_file):
        try:
            self.df = pd.read_csv(csv_file)
            self._cleanup_data()
        except Exception as e:
            raise Exception(f"couldn't load data: {e}")
        
    def _cleanup_data(self):
        original_count = len(self.df)
        
        self.df = self.df.dropna(subset=['job_description_text'])
        self.df['job_description_text'] = self.df['job_description_text'].astype(str)
        self.df['seniority_level'] = self.df['seniority_level'].fillna('unknown').astype(str).str.strip().str.lower()
        
        # categorize seniority
        self.df['seniority_category'] = self.df['seniority_level'].apply(self._get_seniority_bucket)
        
    def _get_seniority_bucket(self, level):
        # check if it's a junior role
        if pd.isna(level) or level == 'unknown':
            return 'unknown'
            
        level = str(level).lower()
        
        entry_words = ['entry', 'junior', 'intern', 'associate', 'fresher', 'trainee', 'graduate']
        senior_words = ['senior', 'lead', 'principal', 'manager', 'director', 'head', 'chief', 'architect']
        
        if any(word in level for word in entry_words):
            return 'entry'
        elif any(word in level for word in senior_words):
            return 'senior'
        else:
            return 'mid'
    
    def extract_skills_fast(self, text):
        # hardcoded patterns for speed
        skill_regex = {
            'python': r'\b(?:python)\b',
            'r': r'\b(?:r)\b(?!\s*(?:&|and))',
            'sql': r'\b(?:sql|mysql|postgresql|sqlite)\b',
            'tensorflow': r'\b(?:tensorflow|tf)\b',
            'pytorch': r'\b(?:pytorch)\b',
            'scikit-learn': r'\b(?:scikit-learn|sklearn)\b',
            'pandas': r'\b(?:pandas)\b',
            'numpy': r'\b(?:numpy)\b',
            'matplotlib': r'\b(?:matplotlib)\b',
            'aws': r'\b(?:aws|amazon web services)\b',
            'azure': r'\b(?:azure|microsoft azure)\b',
            'gcp': r'\b(?:gcp|google cloud)\b',
            'docker': r'\b(?:docker)\b',
            'kubernetes': r'\b(?:kubernetes|k8s)\b',
            'git': r'\b(?:git|github|gitlab)\b',
            'machine learning': r'\b(?:machine learning|ml)\b',
            'deep learning': r'\b(?:deep learning|dl)\b',
            'nlp': r'\b(?:nlp|natural language processing)\b',
            'computer vision': r'\b(?:computer vision|cv)\b',
            'java': r'\b(?:java)\b',
            'scala': r'\b(?:scala)\b',
            'spark': r'\b(?:spark|apache spark)\b',
            'tableau': r'\b(?:tableau)\b',
            'javascript': r'\b(?:javascript|js)\b',
            'react': r'\b(?:react|reactjs)\b',
            'angular': r'\b(?:angular)\b',
            'node.js': r'\b(?:node\.?js|nodejs)\b',
            'mongodb': r'\b(?:mongodb|mongo)\b',
            'kafka': r'\b(?:kafka|apache kafka)\b',
            'jupyter': r'\b(?:jupyter)\b',
            'linux': r'\b(?:linux|ubuntu|centos)\b',
            'api': r'\b(?:api|rest api|restful)\b',
            'microservices': r'\b(?:microservices)\b',
            'agile': r'\b(?:agile|scrum)\b',
            'devops': r'\b(?:devops)\b',
            'jenkins': r'\b(?:jenkins)\b',
            'terraform': r'\b(?:terraform)\b'
        }
        
        text_lower = text.lower()
        found = []
        
        for skill, pattern in skill_regex.items():
            if re.search(pattern, text_lower):
                found.append(skill)
                
        return found
    
    def compare_skills_by_level(self):
        
        entry_skills = []
        senior_skills = []
        
        for idx, row in self.df.iterrows():
            skills = self.extract_skills_fast(row['job_description_text'])
            if row['seniority_category'] == 'entry':
                entry_skills.extend([skill.lower().strip() for skill in skills])
            elif row['seniority_category'] == 'senior':
                senior_skills.extend([skill.lower().strip() for skill in skills])
        
        entry_counts = Counter(entry_skills)
        senior_counts = Counter(senior_skills)
        
        return {
            'entry_skills': dict(entry_counts.most_common(15)),
            'senior_skills': dict(senior_counts.most_common(15)),
            'entry_jobs': len(self.df[self.df['seniority_category'] == 'entry']),
            'senior_jobs': len(self.df[self.df['seniority_category'] == 'senior']),
            'entry_unique': len(set(entry_skills)),
            'senior_unique': len(set(senior_skills))
        }
    
    def get_top_demanded_skills(self):
        # top 3 most wanted skills overall
        
        all_skills = []
        for idx, row in self.df.iterrows():
            skills = self.extract_skills_fast(row['job_description_text'])
            all_skills.extend([skill.lower().strip() for skill in skills])
        
        skill_counts = Counter(all_skills)
        return dict(skill_counts.most_common(3))
    
    def find_patterns(self):
        # discover some interesting stuff
        patterns = {}
        
        # where are most jobs?
        location_dist = self.df['company_address_locality'].value_counts().head(10)
        patterns['top_locations'] = dict(location_dist)
        
        # company types
        ai_companies = self.df[self.df['company_name'].str.contains('AI|ai|Artificial Intelligence', na=False, case=False)]
        big_tech = self.df[self.df['company_name'].str.contains('Google|Amazon|Microsoft|Apple|Meta|Netflix|IBM', na=False, case=False)]
        
        patterns['company_breakdown'] = {
            'ai_companies': len(ai_companies),
            'big_tech': len(big_tech),
            'total_companies': self.df['company_name'].nunique()
        }
        
        # job titles analysis
        titles = self.df['job_title'].dropna().astype(str)
        title_categories = self._categorize_titles(titles)
        patterns['job_categories'] = title_categories
        
        patterns['dataset_overview'] = {
            'total_jobs': len(self.df),
            'unique_locations': self.df['company_address_locality'].nunique(),
            'unique_companies': self.df['company_name'].nunique(),
            'level_breakdown': dict(self.df['seniority_category'].value_counts())
        }
        
        return patterns
    
    def _categorize_titles(self, job_titles):
        # simple title categorization
        categories = {
            'Data Scientist': 0,
            'ML Engineer': 0,
            'Software Engineer': 0,
            'Data Engineer': 0,
            'AI Engineer': 0,
            'Research Scientist': 0,
            'Product Manager': 0,
            'Other': 0
        }
        
        for title in job_titles:
            title_lower = title.lower()
            if 'data scientist' in title_lower:
                categories['Data Scientist'] += 1
            elif 'machine learning' in title_lower or 'ml engineer' in title_lower:
                categories['ML Engineer'] += 1
            elif 'software engineer' in title_lower and 'data' not in title_lower:
                categories['Software Engineer'] += 1
            elif 'data engineer' in title_lower:
                categories['Data Engineer'] += 1
            elif 'ai engineer' in title_lower:
                categories['AI Engineer'] += 1
            elif 'research scientist' in title_lower:
                categories['Research Scientist'] += 1
            elif 'product manager' in title_lower:
                categories['Product Manager'] += 1
            else:
                categories['Other'] += 1
        
        return {k: v for k, v in categories.items() if v > 0}
    
    def create_charts(self):
        # make some charts
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # skills comparison chart
        skill_data = self.compare_skills_by_level()
        
        all_skills = set(list(skill_data['entry_skills'].keys()) + list(skill_data['senior_skills'].keys()))
        comparison = []
        
        for skill in list(all_skills)[:10]:  
            entry_count = skill_data['entry_skills'].get(skill, 0)
            senior_count = skill_data['senior_skills'].get(skill, 0)
            comparison.append({
                'skill': skill,
                'entry': entry_count,
                'senior': senior_count
            })
        
        comp_df = pd.DataFrame(comparison)
        comp_df = comp_df.sort_values('entry', ascending=False)
        
        x = np.arange(len(comp_df))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, comp_df['entry'], width, label='Entry Level', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, comp_df['senior'], width, label='Senior Level', alpha=0.8, color='lightcoral')
        
        axes[0, 0].set_xlabel('Skills')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Skills: Entry vs Senior')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(comp_df['skill'], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # locations chart
        pattern_data = self.find_patterns()
        locations = list(pattern_data['top_locations'].keys())[:8]
        counts = list(pattern_data['top_locations'].values())[:8]
        
        bars = axes[0, 1].barh(locations, counts, color='mediumseagreen')
        axes[0, 1].set_xlabel('Job Count')
        axes[0, 1].set_title('Top Job Locations')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0, 1].text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                           f'{int(width)}', ha='left', va='center')
        
        # top skills pie
        top_skills = self.get_top_demanded_skills()
        colors = ['gold', 'silver', '#CD7F32']  
        wedges, texts, autotexts = axes[0, 2].pie(top_skills.values(), labels=top_skills.keys(), 
                                                 autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 2].set_title('Top 3 Skills Overall')
        
        # seniority distribution
        seniority_dist = self.df['seniority_category'].value_counts()
        bars = axes[1, 0].bar(seniority_dist.index, seniority_dist.values, 
                             color=['lightblue', 'lightgreen', 'lightyellow', 'lightpink'])
        axes[1, 0].set_xlabel('Level')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Jobs by Seniority')
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom')
        
        # company types
        company_stats = pattern_data['company_breakdown']
        categories = ['AI Companies', 'Big Tech', 'Others']
        values = [
            company_stats['ai_companies'],
            company_stats['big_tech'],
            company_stats['total_companies'] - company_stats['ai_companies'] - company_stats['big_tech']
        ]
        
        bars = axes[1, 1].bar(categories, values, color=['purple', 'orange', 'gray'])
        axes[1, 1].set_ylabel('Company Count')
        axes[1, 1].set_title('Company Types')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom')
        
        # job title categories
        title_cats = pattern_data['job_categories']
        if title_cats:
            cats = list(title_cats.keys())[:5]
            counts = list(title_cats.values())[:5]
            
            bars = axes[1, 2].barh(cats, counts, color='lightsteelblue')
            axes[1, 2].set_xlabel('Count')
            axes[1, 2].set_title('Job Categories')
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[1, 2].text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                               f'{int(width)}', ha='left', va='center')
        else:
            axes[1, 2].text(0.5, 0.5, 'No title data', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Job Categories')
        
        plt.tight_layout()
        
        output_file = 'job_market_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_file


class SkillClassifier:
    # main class for skill detection and trend analysis
    
    def __init__(self, groq_client):
        self.client = groq_client
        self.parser = TextParser(groq_client)
    
    def analyze_job_description(self, job_desc):
        # extract and classify skills from a job description
        skills = self.parser.get_skills_from_text(job_desc)
        
        if not skills:
            return []
        
        results = []
        
        for skill in set(skills):  # remove dupes
            if skill.strip():
                category, score = self.parser.classify_skill(skill)
                results.append(SkillData(
                    skill=skill.strip(),
                    category=category,
                    trend_score=round(score, 2)
                ))
        
        # sort by emerging first, then by score
        results.sort(key=lambda x: (x.category == 'established', -x.trend_score))
        
        return results


# setup FastAPI
app = FastAPI(
    title="Job Market Analysis API",
    description="Analyze job descriptions and detect skill trends",
    version="1.0.0"
)

# initialize LLM components
try:
    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key:
        raise Exception("missing GROQ_API_KEY")
    
    client = Groq(api_key=groq_key)
    classifier = SkillClassifier(client)
except Exception as e:
    print(f"LLM setup failed: {e}")
    classifier = None

@app.get("/", response_model=StatusResponse)
async def home():
    return StatusResponse(
        status="running", 
        message="Job Market Analysis API is up. Check /docs for endpoints."
    )

@app.get("/health", response_model=StatusResponse)
async def health():
    llm_status = "working" if classifier else "unavailable"
    return StatusResponse(
        status="healthy", 
        message=f"API is running. LLM: {llm_status}"
    )

@app.post("/detect-skills", response_model=SkillResponse)
async def extract_skills(request: JDInput):
    # skill detection endpoint
    try:
        if not request.job_description.strip():
            raise HTTPException(status_code=400, detail="job description is empty")
        
        if classifier is None:
            raise HTTPException(status_code=500, detail="LLM not available - check GROQ_API_KEY")
        
        detected = classifier.analyze_job_description(request.job_description)
        
        return SkillResponse(
            detected_skills=detected,
            total_skills=len(detected)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"skill detection error: {str(e)}")

def convert_numpy_types(obj):
    # helper to convert numpy types to regular python types
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    else:
        return obj

@app.post("/analyze-dataset")
async def dataset_analysis(request: DatasetRequest):
    # dataset analysis endpoint
    try:
        if not os.path.exists(request.csv_path):
            raise HTTPException(status_code=404, detail=f"file not found: {request.csv_path}")
        
        analyzer = DataAnalyzer(request.csv_path)
        skill_comparison = analyzer.compare_skills_by_level()
        top_skills = analyzer.get_top_demanded_skills()
        patterns = analyzer.find_patterns()
        chart_file = analyzer.create_charts()
        
        response = {
            "summary": {
                "description": "comprehensive job market analysis",
                "method": "regex pattern matching for speed",
                "performance": "optimized for large datasets"
            },
            "skill_comparison": convert_numpy_types(skill_comparison),
            "top_skills": convert_numpy_types(top_skills),
            "patterns": convert_numpy_types(patterns),
            "dataset_stats": {
                "total_jobs": int(len(analyzer.df)),
                "locations": int(analyzer.df['company_address_locality'].nunique()),
                "companies": int(analyzer.df['company_name'].nunique()),
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "visualization": {
                "created": True,
                "path": chart_file,
                "note": "charts saved successfully"
            }
        }
        return JSONResponse(content=convert_numpy_types(response))
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"analysis error: {str(e)}")

@app.post("/test-llm")
async def test_llm():
    # quick test to check if LLM is working
    if not classifier:
        raise HTTPException(status_code=500, detail="LLM not available")
    
    test_jd = """
    Senior ML Engineer needed with skills in:
    - Python and PyTorch
    - Large Language Models and transformers  
    - LangChain for AI apps
    - Vector databases like Pinecone
    - MLOps with Docker/Kubernetes
    - Diffusion models and generative AI
    - AWS/GCP cloud platforms
    """
    
    try:
        results = classifier.analyze_job_description(test_jd)
        return {
            "test": "passed",
            "skills_found": len(results),
            "sample": [
                {
                    "skill": s.skill,
                    "category": s.category,
                    "score": s.trend_score
                } for s in results[:5]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"test failed: {str(e)}")


if __name__ == '__main__':
    
    # quick LLM test on startup
    if classifier:
        sample = "Python developer with experience in TensorFlow, LangChain, and vector databases"
        try:
            test_results = classifier.analyze_job_description(sample)
        except Exception as e:
            print(f"LLM test failed: {e}")
    else:
        print("LLM not available - check GROQ_API_KEY")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)