from fastapi import FastAPI, Depends, Request, HTTPException, Form, File, UploadFile, Header
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta, datetime
from jose import JWTError, jwt
from passlib.context import CryptContext
import os, requests, json, shutil, openai, uuid, base64
from dotenv import load_dotenv  
import xml.etree.ElementTree as ET
from backend.database import engine, get_db, SessionLocal
from backend.models import Base, User, Chat, Message, Stage
from backend.schemas import UserCreate
from pydantic import BaseModel
from typing import Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from io import BytesIO


# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

# Setup paths for frontend
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Jinja2 template setup
templates = Jinja2Templates(directory=FRONTEND_DIR)

# Security settings
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

load_dotenv()

PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
ELSEVIER_API_KEY = os.getenv("ELSEVIER_API_KEY")
PMC_API_KEY = os.getenv("PMC_API_KEY")

# Debugging API Key Fetching
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("⚠️ ERROR: OPENAI_API_KEY is missing! Check your .env file.")

# Initialize OpenAI Client
from openai import OpenAI
client = OpenAI(api_key=api_key)
print("✅ OpenAI API Key Loaded Successfully!")

# Add fallback model configuration
DEFAULT_MODEL = "gpt-4"
FALLBACK_MODEL = "gpt-3.5-turbo"
LARGE_CONTEXT_MODEL = "gpt-4-turbo"  # Model with larger context window

methodology_data = None
results_data = None

UPLOAD_FOLDER = "backend/uploads"  # Store images in backend
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure folder exists

@app.post("/upload-results")
async def upload_results(
    chat_id: int = Form(...),
    outcome_variables_text: str = Form(None),
    exposure_variables_text: Optional[str] = Form(None),
    predictors_text: Optional[str] = Form(None),
    potential_confounders_text: Optional[str] = Form(None),
    effect_modifiers_text: Optional[str] = Form(None),
    outcome_variables_image: UploadFile = File(None),
    exposure_variables_image: Optional[UploadFile] = File(None),
    predictors_image: Optional[UploadFile] = File(None),
    potential_confounders_image: Optional[UploadFile] = File(None),
    effect_modifiers_image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    # Save uploaded images
    saved_files = []
    for image in [outcome_variables_image, exposure_variables_image, predictors_image, potential_confounders_image, effect_modifiers_image]:
        if image:
            file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{image.filename}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            saved_files.append(file_path)
    
    # Only process images if there are any uploaded
    extracted_text = ""
    if saved_files:
        extracted_text = process_with_gpt4_vision(saved_files, {
            "Outcome Variables": outcome_variables_text,
            "Exposure Variables": exposure_variables_text,
            "Predictors": predictors_text,
            "Potential Confounders": potential_confounders_text,
            "Effect Modifiers": effect_modifiers_text
        })
    
    # Combine form data and extracted text
    structured_results_message = f"""
    RESULTS
    1. Variables:
    - Outcome Variables: {outcome_variables_text}
    - Exposure Variables: {exposure_variables_text}
    - Predictors: {predictors_text}
    - Potential Confounders: {potential_confounders_text}
    - Effect Modifiers: {effect_modifiers_text}
    """

    # Only add image data section if there are images
    if extracted_text:
        structured_results_message += f"""

    2. Extracted Image Data:
    {extracted_text}
    """

    # Process with GPT and store in database
    gpt_response = process_message("Results", structured_results_message, chat_id, db)
    new_message = Message(
        chat_id=chat_id,
        role="bot - Results",
        text=gpt_response,
        timestamp=datetime.utcnow()
    )
    db.add(new_message)
    db.commit() 

    return JSONResponse({"response": gpt_response})


def encode_image(image_path):
    """Convert image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def process_with_gpt4_vision(image_paths, text_data):
    """Send images and text to GPT-4 Vision for analysis."""
    try:
        messages = [{"role": "system", "content": """You are an AI assistant analyzing research data images.
        For each image:
        1. Describe the type of data visualization (table, graph, chart, etc.)
        2. Extract all numerical values and their labels
        3. Identify any trends, patterns, or significant findings
        4. List any statistical values (p-values, confidence intervals, etc.)
        5. Note any anomalies or outliers
        Format your response in a structured way for each image."""}]

        # Add text-based variables to messages
        for key, value in text_data.items():
            if value:
                messages.append({"role": "user", "content": f"{key}: {value}"})

        # Add images in Base64 format with specific instructions
        image_contents = []
        for path in image_paths:
            try:
                base64_image = encode_image(path)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                })
            except Exception as e:
                print(f"❌ Error encoding {path}: {e}")

        if image_contents:
            messages.append({
                "role": "user", 
                "content": [
                    "Please analyze each image and provide a detailed breakdown of the research data shown. Include all numerical values, statistical findings, and any significant patterns or trends.",
                    *image_contents
                ]
            })

        # Call GPT-4 Vision with gpt-4o model
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        gpt_output = response.choices[0].message.content if response else "No response from OpenAI."
        print(f"✅ GPT Vision Response: {gpt_output}")
        
        # Format the structured results message with enhanced styling
        structured_results = f"""
        <div class="results-container">
            <h2 class="results-title">Research Analysis Results</h2>
            
            <div class="variables-section">
                <h3>Variables Analysis</h3>
                <ul class="variables-list">
                    <li><strong>Outcome Variables:</strong> {text_data.get('Outcome Variables', 'N/A')}</li>
                    <li><strong>Exposure Variables:</strong> {text_data.get('Exposure Variables', 'N/A')}</li>
                    <li><strong>Predictors:</strong> {text_data.get('Predictors', 'N/A')}</li>
                    <li><strong>Potential Confounders:</strong> {text_data.get('Potential Confounders', 'N/A')}</li>
                    <li><strong>Effect Modifiers:</strong> {text_data.get('Effect Modifiers', 'N/A')}</li>
                </ul>
            </div>

            <div class="image-analysis-section">
                <h3>Image Analysis Results</h3>
                <div class="analysis-content">
                    {gpt_output}
                </div>
            </div>
        </div>
        """
        return structured_results

    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return f"Error processing images: {str(e)}"


@app.post("/submit-methodology")
async def submit_methodology(request: Request):
    global methodology_data
    methodology_data = await request.json()
    return {"message": "Methodology Data Received"}

@app.post("/submit-results")
async def submit_results(request: Request):
    global results_data
    results_data = await request.json()

    print(f"✅ Results Data Stored: {results_data}")  # Debugging output

    return {"message": "Results Data Received"}


def can_run_gpt(gpt_name):
    if gpt_name == "methodology":
        return methodology_data is not None
    if gpt_name == "results":
        return methodology_data is not None and results_data is not None
    return True


def extract_field(data, keys, default="N/A"):
    """Extract the first available field from multiple key options."""
    for key in keys:
        if key in data and data[key]:
            return data[key]
    return default

def extract_authors(authors_list):
    """Extract authors dynamically by checking multiple possible keys."""
    return ", ".join([author.get("name", "N/A") for author in authors_list]) if authors_list else "N/A"


def fetch_research_articles(query):
    """
    Enhanced function to fetch research articles from multiple sources:
    - PubMed (NCBI)
    - Europe PMC
    - OpenAlex
    - Semantic Scholar
    - arXiv
    - Core
    """
    results = []

    # Helper function to standardize article format
    def standardize_article(article, source):
        return {
            "Source": source,
            "Title": article.get("Title", "N/A"),
            "Authors": article.get("Authors", "N/A"),
            "Published Date": article.get("Published Date", "N/A"),
            "Journal": article.get("Journal", "N/A"),
            "Abstract": article.get("Abstract", ""),
            "DOI": article.get("DOI", "N/A"),
            "URL": article.get("URL", "N/A")
        }

    # 1. PubMed implementation
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 10,
            "api_key": PUBMED_API_KEY,
            "email": "abhi42cy7@gmail.com",
            "tool": "research_assistant"
        }
        
        search_response = requests.get(search_url, params=search_params)
        search_response.raise_for_status()
        pmids = search_response.json().get("esearchresult", {}).get("idlist", [])

        for pmid in pmids:
            try:
                details_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json&api_key={PUBMED_API_KEY}"
                abstract_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml&api_key={PUBMED_API_KEY}"
                
                details_response = requests.get(details_url)
                abstract_response = requests.get(abstract_url)
                
                if details_response.ok and abstract_response.ok:
                    details = details_response.json()["result"][pmid]
                    abstract_root = ET.fromstring(abstract_response.text)
                    abstract = " ".join(elem.text for elem in abstract_root.findall(".//Abstract/AbstractText") if elem.text)
                    
                    article = {
                        "Title": details.get("title", ""),
                        "Authors": ", ".join(author.get("name", "") for author in details.get("authors", [])),
                        "Published Date": details.get("pubdate", ""),
                        "Journal": details.get("fulljournalname", ""),
                        "Abstract": abstract,
                        "URL": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
                        "DOI": next((id_info["value"] for id_info in details.get("articleids", []) if id_info["idtype"] == "doi"), "N/A")
                    }
                    results.append(standardize_article(article, "PubMed"))
            except Exception as e:
                print(f"❌ Error processing PubMed article {pmid}: {str(e)}")
    except Exception as e:
        print(f"❌ PubMed API Error: {str(e)}")

    # 2. Europe PMC
    try:
        europe_pmc_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": query,
            "format": "json",
            "pageSize": 10
        }
        response = requests.get(europe_pmc_url, params=params)
        if response.ok:
            data = response.json()
            for result in data.get("resultList", {}).get("result", []):
                article = {
                    "Title": result.get("title", ""),
                    "Authors": result.get("authorString", ""),
                    "Published Date": result.get("firstPublicationDate", ""),
                    "Journal": result.get("journalTitle", ""),
                    "Abstract": result.get("abstractText", ""),
                    "DOI": result.get("doi", "N/A"),
                    "URL": f"https://europepmc.org/article/{result.get('source')}/{result.get('id')}"
                }
                results.append(standardize_article(article, "Europe PMC"))
    except Exception as e:
        print(f"❌ Europe PMC API Error: {str(e)}")

    # 3. OpenAlex
    try:
        openalex_url = "https://api.openalex.org/works"
        params = {
            "filter": f"title.search:{query}",
            "per-page": 10
        }
        response = requests.get(openalex_url, params=params)
        if response.ok:
            data = response.json()
            for work in data.get("results", []):
                article = {
                    "Title": work.get("title", ""),
                    "Authors": ", ".join(author.get("author", {}).get("display_name", "") for author in work.get("authorships", [])),
                    "Published Date": work.get("publication_date", ""),
                    "Journal": work.get("primary_location", {}).get("source", {}).get("display_name", ""),
                    "Abstract": work.get("abstract", ""),
                    "DOI": work.get("doi", "N/A"),
                    "URL": work.get("primary_location", {}).get("landing_page_url", "N/A")
                }
                results.append(standardize_article(article, "OpenAlex"))
    except Exception as e:
        print(f"❌ OpenAlex API Error: {str(e)}")

    # 4. arXiv
    try:
        arxiv_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "max_results": 10
        }
        response = requests.get(arxiv_url, params=params)
        if response.ok:
            root = ET.fromstring(response.text)
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                article = {
                    "Title": entry.find(".//{http://www.w3.org/2005/Atom}title").text,
                    "Authors": ", ".join(author.find(".//{http://www.w3.org/2005/Atom}name").text 
                                       for author in entry.findall(".//{http://www.w3.org/2005/Atom}author")),
                    "Published Date": entry.find(".//{http://www.w3.org/2005/Atom}published").text.split("T")[0],
                    "Journal": "arXiv",
                    "Abstract": entry.find(".//{http://www.w3.org/2005/Atom}summary").text,
                    "URL": entry.find(".//{http://www.w3.org/2005/Atom}id").text,
                    "DOI": "N/A"
                }
                results.append(standardize_article(article, "arXiv"))
    except Exception as e:
        print(f"❌ arXiv API Error: {str(e)}")

    # Sort and remove duplicates
    seen_titles = set()
    unique_results = []
    
    # First, sort by relevance (presence of query terms in title/abstract)
    def relevance_score(article):
        score = 0
        query_terms = query.lower().split()
        title = article["Title"].lower()
        abstract = article["Abstract"].lower()
        
        # Check title (higher weight)
        for term in query_terms:
            if term in title:
                score += 3
            if term in abstract:
                score += 1
                
        # Boost score for recent articles
        try:
            date = datetime.strptime(article["Published Date"], "%Y-%m-%d")
            years_old = (datetime.now() - date).days / 365
            score += max(0, 5 - years_old)  # Boost for newer articles
        except:
            pass
            
        return score

    # Sort by relevance score
    sorted_results = sorted(results, key=relevance_score, reverse=True)
    
    # Remove duplicates while preserving order
    for article in sorted_results:
        title = article["Title"].lower()
        if title not in seen_titles:
            seen_titles.add(title)
            unique_results.append(article)

    return unique_results[:20]  # Return top 20 most relevant results

# Example usage:
if __name__ == "__main__":
    PUBMED_API_KEY = "your_pubmed_api_key"
    ELSEVIER_API_KEY = "your_elsevier_api_key"

    query = "Comparison of accuracy of detection of high-grade prostate cancer using DWI on MRI versus prostate biopsy"
    articles = fetch_research_articles(query)
    print(json.dumps(articles, indent=2))


def get_openai_response(messages, max_retries=3):
    """Helper function to make OpenAI API calls with fallback and retry logic"""
    current_model = DEFAULT_MODEL
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=current_model,
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  API error: {error_msg}")
            
            # Check for different error types and apply appropriate fallback
            if "insufficient_quota" in error_msg and current_model == DEFAULT_MODEL:
                print(f"⚠️ Quota exceeded for {current_model}, falling back to {FALLBACK_MODEL}")
                current_model = FALLBACK_MODEL
                continue
            elif "context_length_exceeded" in error_msg:
                # Switch to a model with larger context window
                print(f"⚠️ Context length exceeded for {current_model}, falling back to {LARGE_CONTEXT_MODEL}")
                current_model = LARGE_CONTEXT_MODEL
                continue
            elif attempt < max_retries - 1:
                print(f"⚠️ API error, retrying... ({attempt + 1}/{max_retries})")
                continue
            else:
                print(f"❌ All retries failed: {error_msg}")
                raise

def process_message(gpt_name, user_message, chat_id, db: Session):
    """Calls the correct GPT model based on the stage, with structured prompts and dependencies."""
    try:
        dependencies = {
            "Introduction": [],  
            "Review": ["Introduction"],
            "Methodology": ["Introduction", "Review"],
            "Results": ["Introduction", "Review", "Methodology"],  
            "Discussion": ["Introduction", "Review", "Methodology", "Results"],  
            "References": ["Review", "Methodology", "Results", "Discussion"],
            "Conclusion": ["Introduction", "Review", "Methodology", "Results", "Discussion", "References"],  
            "Abstract": ["Introduction", "Review", "Methodology", "Results", "Discussion", "References", "Conclusion"]  
        }

        # Verify all required dependencies are present
        if gpt_name in dependencies:
            required_stages = dependencies[gpt_name]
            messages = db.query(Message).filter(
                Message.chat_id == chat_id,
                Message.role.in_([f"bot - {stage}" for stage in required_stages])
            ).order_by(Message.timestamp).all()

            missing_stages = [stage for stage in required_stages 
                            if not any(msg.role == f"bot - {stage}" for msg in messages)]
            
            if missing_stages:
                return f"Error: Missing required stages: {', '.join(missing_stages)}"

            # Get the most recent message for each stage
            stage_messages = {}
            for msg in messages:
                stage = msg.role.replace("bot - ", "")
                stage_messages[stage] = msg.text

    except Exception as e:
        print(f"❌ Error in process_message: {str(e)}")
        return f"Error processing message: {str(e)}"

    try:
        # Special handling for Review stage
        if gpt_name == "Review":
            print("✅ Fetching research articles for Review stage...")
            # Fetch relevant research articles
            research_articles = fetch_research_articles(user_message)
            
            if not research_articles:
                print("⚠️ No research articles found!")
                return "Error: Could not find relevant research articles. Please try a different search query."
            
            # Format articles for GPT with more detailed information
            formatted_articles = "\n\nRELEVANT RESEARCH ARTICLES:\n"
            for i, article in enumerate(research_articles[:10], 1):
                formatted_articles += f"\n{i}. Title: {article['Title']}\n"
                formatted_articles += f"   Authors: {article['Authors']}\n"
                formatted_articles += f"   Journal: {article['Journal']}\n"
                formatted_articles += f"   Published Date: {article['Published Date']}\n"
                if article.get('Abstract'):
                    formatted_articles += f"   Abstract: {article['Abstract']}\n"
                formatted_articles += f"   Source: {article['Source']}\n"
                formatted_articles += f"   Citation Link: {article.get('DOI', 'N/A') if article.get('DOI') != 'N/A' else article.get('URL')}\n"
                formatted_articles += "\n---\n"

            # Enhanced prompt for Review stage with emphasis on real papers and citations
            review_prompt = f"""
            Previous Introduction:
            {stage_messages.get('Introduction', 'No introduction available')}

            Based on the following research topic and the provided REAL academic articles from multiple sources:
            Topic: {user_message}

            {formatted_articles}

            Please provide a comprehensive literature review that:
            1. Analyzes and synthesizes these specific research articles
            2. Groups findings by themes or research areas
            3. Compares methodologies across the provided studies
            4. Evaluates the strength of evidence from each paper
            5. Identifies gaps and contradictions in these papers
            6. Discusses the relevance of these specific findings to the research objectives

            IMPORTANT FORMATTING REQUIREMENTS:
            1. ALWAYS cite papers using this exact HTML format: 
               - First citation: (Author et al., Year <a href="full_citation_link" target="_blank" class="paper-link">[View Paper]</a>)
               - Subsequent citations: (Author et al., Year)
            2. At the end of each major section, list the full citation links as:
               <div class="section-references">
               Papers cited in this section:
               - Author et al., Year: <a href="full_citation_link" target="_blank" class="paper-link">View Paper</a>
               </div>
            3. Include a "References" section at the end with all papers as:
               <div class="references">
               References:
               - Author et al., Year: <a href="full_citation_link" target="_blank" class="paper-link">View Paper</a>
               </div>
            4. When discussing methodology or findings, explicitly state which paper(s) you're referencing
            5. For each key finding, cite ALL relevant papers that support or contradict it

            Structure the review as follows:
            1. Overview of Current Research
            2. Key Themes and Findings
            3. Methodological Approaches
            4. Gaps and Future Directions
            5. Relevance to Current Study
            6. References

            Remember: ONLY use the provided research articles - do not reference any papers not listed above.
            """

            # Use the enhanced prompt instead of user_message
            user_message = review_prompt

        # Define structured prompts for each stage
        prompts = {
            "Introduction": """
            Write a comprehensive introduction for a research paper on the following topic:
            {user_message}

            The introduction should:
            1. Provide background and context
            2. State the research problem
            3. Explain the significance of the study
            4. Present the research objectives
            5. Outline the paper structure
            """,
            
            "Review": """
            Previous Introduction:
            {introduction}

            Based on the following research topic and provided articles from multiple academic sources:
            {user_message}

            Please provide a comprehensive literature review that:
            1. Analyzes and synthesizes the provided research articles
            2. Groups findings by themes or research areas
            3. Compares methodologies across studies
            4. Evaluates the strength of evidence for each finding
            5. Identifies gaps and contradictions in the literature
            6. Highlights the most recent developments (within last 2-3 years)
            7. Discusses the relevance of findings to your research objectives
            8. Suggests potential directions for further investigation

            Structure the review as follows:
            1. Overview of Current Research
            2. Key Themes and Findings
            3. Methodological Approaches
            4. Gaps and Future Directions
            5. Relevance to Current Study

            For each major finding, include:
            - Supporting evidence from multiple sources
            - Any contradicting evidence
            - Methodological strengths/limitations
            - Relevance to your research objectives
            """,
            
            "Methodology": """
            Previous sections:
            Introduction: {introduction}
            Review: {review}

            Design a detailed methodology for researching:
            {user_message}

            Include:
            1. Study design and rationale
            2. Participant selection criteria
            3. Data collection methods
            4. Analysis procedures
            5. Ethical considerations
            6. Limitations and potential biases
            """,
            
            "Results": """
            Previous sections:
            Methodology: {methodology}

            Analyze the following research results:
            {user_message}

            Provide:
            1. Summary of key findings
            2. Statistical analysis
            3. Data visualization recommendations
            4. Patterns and trends
            5. Unexpected findings
            """,
            
            "Discussion": """
            Previous sections:
            Results: {results}
            Methodology: {methodology}

            Discuss the implications of these findings:
            {user_message}

            Include:
            1. Interpretation of results
            2. Comparison with previous studies
            3. Theoretical and practical implications
            4. Study limitations
            5. Future research directions
            """,
            
            "References": """
            Based on the following sections:
            Review: {review}
            Methodology: {methodology}
            Results: {results}
            Discussion: {discussion}

            Generate appropriate references and citations for all sources mentioned.
            Ensure proper academic citation format.
            """,
            
            "Conclusion": """
            Based on all previous sections:
            Introduction: {introduction}
            Results: {results}
            Discussion: {discussion}
            References: {references}

            Provide a comprehensive conclusion that:
            1. Summarizes key findings
            2. Addresses research objectives
            3. Discusses implications
            4. Suggests future directions
            """,
            
            "Abstract": """
            Based on all completed sections:
            Introduction: {introduction}
            Methodology: {methodology}
            Results: {results}
            Discussion: {discussion}
            Conclusion: {conclusion}

            Generate a comprehensive abstract that:
            1. Summarizes the entire paper
            2. Highlights key findings
            3. States main conclusions
            4. Follows standard abstract structure
            """
        }

        # Prepare the prompt based on GPT type
        if gpt_name in prompts:
            # Create a format dictionary with all possible previous responses
            format_dict = {
                "user_message": user_message,
                "introduction": stage_messages.get("Introduction", "No introduction available"),
                "review": stage_messages.get("Review", "No review available"),
                "methodology": stage_messages.get("Methodology", "No methodology available"),
                "results": stage_messages.get("Results", "No results available"),
                "discussion": stage_messages.get("Discussion", "No discussion available"),
                "references": stage_messages.get("References", "No references available"),
                "conclusion": stage_messages.get("Conclusion", "No conclusion available")
            }
            
            # Format the prompt with available values
            prompt = prompts[gpt_name].format(**format_dict)
        else:
            prompt = f"""
            {gpt_name.upper()} SECTION
            Topic: {user_message}

            Previous Information:
            {stage_messages}

            Please provide a detailed and well-structured response following academic writing standards.
            """

        # Prepare system message based on the stage
        system_message = {
            "role": "system",
            "content": f"You are an expert academic research assistant specialized in {gpt_name} sections."
        }

        # Prepare user message with context
        messages = [
            system_message,
            {"role": "user", "content": prompt}
        ]

        try:
            # Use the helper function with fallback and retry logic
            response_text = get_openai_response(messages)
            return response_text
        except Exception as api_error:
            error_msg = str(api_error)
            print(f"❌ OpenAI API Error: {error_msg}")
            if "insufficient_quota" in error_msg:
                return "Error: API quota exceeded. Please try again later or contact support."
            return f"Error generating response: {error_msg}"

    except Exception as e:
        print(f"❌ Error in process_message: {str(e)}")
        return f"Error processing message: {str(e)}"

def verify_token(token: str, db: Session):
    """Decode JWT and check if user exists in DB."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            return False  # Invalid token
        user = db.query(User).filter(User.username == username).first()
        return user is not None
    except JWTError:
        return False

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_token_from_header(authorization: str = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "")
    return None

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    hashed_password = pwd_context.hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    return {"message": "User registered successfully"}

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user or not pwd_context.verify(password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_access_token(data={"sub": db_user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/logout")
def logout():
    """Endpoint to handle user logout."""
    return JSONResponse(
        status_code=200,
        content={"message": "Logged out successfully", "redirect": "/"}
    )

@app.get("/")
def show_login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui(request: Request):
    """Simply return the chat UI without authentication."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat/new")
def create_new_chat(db: Session = Depends(get_db)):
    new_chat = Chat(title="New Research Session")
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    stage = Stage(chat_id=new_chat.id, current_stage=1)
    db.add(stage)
    db.commit()
    return {"chat_id": new_chat.id, "message": "New chat session created"}

# ✅ Create a request schema
class ChatRequest(BaseModel):
    chat_id: int
    gpt_name: str
    message: str

@app.post("/chat")
def chat_with_gpt(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        print(f"🔍 Received request: {request}")
        chat_session = db.query(Chat).filter(Chat.id == request.chat_id).first()
        if not chat_session:
            print("❌ Chat session not found!")
            return JSONResponse(
                status_code=404,
                content={"error": "Chat not found", "response": None}
            )

        # Get current stage
        stage = db.query(Stage).filter(Stage.chat_id == request.chat_id).first()
        if not stage:
            stage = Stage(chat_id=request.chat_id, current_stage=1)
            db.add(stage)
            db.commit()
            db.refresh(stage)

        # Get previous messages for context
        previous_messages = db.query(Message).filter(
            Message.chat_id == request.chat_id,
            Message.role.like("bot - %")
        ).order_by(Message.timestamp).all()

        # Build context from previous messages
        context = {}
        for msg in previous_messages:
            stage_name = msg.role.replace("bot - ", "")
            context[stage_name] = msg.text

        # Store Methodology Data if needed
        global methodology_data
        if request.gpt_name == "Methodology":
            methodology_data = request.message
            print("✅ Methodology data stored!")

        # Build structured message based on the stage
        structured_message = request.message
        if request.gpt_name in ["Discussion", "References", "Conclusion", "Abstract"]:
            structured_message = f"""
            Research Topic: {request.message}

            Previous Sections:
            """
            # Add relevant previous sections based on dependencies
            if "Introduction" in context:
                structured_message += f"\nIntroduction:\n{context['Introduction']}\n"
            if "Methodology" in context:
                structured_message += f"\nMethodology:\n{context['Methodology']}\n"
            if "Results" in context:
                structured_message += f"\nResults:\n{context['Results']}\n"
            if request.gpt_name in ["References", "Conclusion", "Abstract"] and "Discussion" in context:
                structured_message += f"\nDiscussion:\n{context['Discussion']}\n"
            if request.gpt_name == "Abstract" and "Conclusion" in context:
                structured_message += f"\nConclusion:\n{context['Conclusion']}\n"

            structured_message += "\nPlease generate the appropriate section based on the above context."

        # Process message and get response
        bot_response = process_message(request.gpt_name, structured_message, request.chat_id, db)
        if not bot_response:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to generate response", "response": None}
            )

        # Save messages to DB
        try:
            # Only save user message for non-auto-generated sections
            if request.gpt_name not in ["Discussion", "References", "Conclusion", "Abstract"]:
                db.add(Message(
                    chat_id=request.chat_id,
                    role="user",
                    text=request.message,
                    stage_id=stage.id,
                    timestamp=datetime.utcnow()
                ))
            
            db.add(Message(
                chat_id=request.chat_id,
                role=f"bot - {request.gpt_name}",
                text=bot_response,
                stage_id=stage.id,
                timestamp=datetime.utcnow()
            ))
            db.commit()
        except Exception as e:
            print(f"❌ Database error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Database error", "response": bot_response}
            )

        print(f"✅ GPT Response for {request.gpt_name}: {bot_response}")
        return JSONResponse(content={"response": bot_response})

    except Exception as e:
        print(f"❌ Unexpected error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "response": None}
        )


@app.get("/chat-history")
def get_chat_history(chat_id: int, db: Session = Depends(get_db)):
    print(f"🔍 Checking chat history for chat_id={chat_id}")  # Debugging
    chat = db.query(Chat).filter(Chat.id == chat_id).first()

    if not chat:
        print("❌ Chat not found in database!")
        return {"history": []}  # ✅ Return an empty history instead of error

    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp).all()

    if not messages:
        print("⚠️ No messages found for this chat!")
        return {"history": []}  # ✅ Avoid frontend errors by returning an empty list

    print(f"✅ Found {len(messages)} messages for chat_id={chat_id}")

    return {"history": [{"role": m.role, "text": m.text} for m in messages]}

@app.get("/download-paper/{chat_id}")
async def download_paper(chat_id: int, format: str = "pdf", db: Session = Depends(get_db)):
    """Generate a formatted research paper as PDF."""
    try:
        # Get all messages in order
        messages = db.query(Message).filter(
            Message.chat_id == chat_id,
            Message.role.like("bot - %")
        ).order_by(Message.timestamp).all()

        # Organize messages by section
        sections = {
            "Abstract": "",
            "Introduction": "",
            "Review": "",
            "Methodology": "",
            "Results": "",
            "Discussion": "",
            "References": "",
            "Conclusion": ""
        }

        for msg in messages:
            section = msg.role.replace("bot - ", "")
            if section in sections:
                sections[section] = msg.text

        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=72)

        # Create styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10
        )
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=12,
            leading=14,
            spaceBefore=6,
            spaceAfter=6
        )

        # Build the PDF content
        elements = []
        
        # Title
        elements.append(Paragraph("RESEARCH PAPER", title_style))
        elements.append(Spacer(1, 12))

        # Abstract
        elements.append(Paragraph("ABSTRACT", heading_style))
        elements.append(Paragraph(sections["Abstract"], body_style))
        elements.append(PageBreak())

        # Main sections
        section_titles = [
            ("1. INTRODUCTION", "Introduction"),
            ("2. LITERATURE REVIEW", "Review"),
            ("3. METHODOLOGY", "Methodology"),
            ("4. RESULTS", "Results"),
            ("5. DISCUSSION", "Discussion"),
            ("6. CONCLUSION", "Conclusion"),
            ("REFERENCES", "References")
        ]

        for title, key in section_titles:
            elements.append(Paragraph(title, heading_style))
            elements.append(Paragraph(sections[key], body_style))
            if key != "References":  # Don't add page break after references
                elements.append(PageBreak())

        # Generate PDF
        doc.build(elements)
        buffer.seek(0)

        # Return the PDF as a downloadable file
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=research_paper.pdf"}
        )

    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

