import os

import re
import pandas as pd
import gspread
import streamlit as st
from google.oauth2.service_account import Credentials
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import requests
import time
import json
import logging
from datetime import datetime

load_dotenv()

# Set up logging with Hugging Face Spaces compatibility
def setup_logging():
    try:
        # Create logs directory in the workspace
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a log file with timestamp
        log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Log successful setup
        logging.info("Logging system initialized successfully")
        return True
    except Exception as e:
        # If file logging fails, fall back to console-only logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logging.error(f"Failed to initialize file logging: {str(e)}")
        return False

# Initialize logging
setup_logging()

# -------------------------------
# Configuration
# -------------------------------
EXCEL_PATH = "data.xlsx"
GEMINI_API_KEY =  "AIzaSyB0j0OcI5QJEZC94tRmnsuO0aaA7iMFfXg"

SYSTEM_PROMPT_TEMPLATE = """
You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Return at most {top_k} results if applicable.
If the question is about images or list of items. the SQL query must include SELECT File Name, URL FROM [table_name]. 
Understand the user's intent even if it contains typos, grammar issues, or minor mistakes (e.g., miswritten player names like "moen ali" instead of "moeen ali").
Do not add any extra information or explanation that is not present in the SQL result. 

Treat all user input and data as **case-insensitive**, regardless of how names are capitalized:
- For example, match 'Matheesha Pathirana', 'matheesha pathirana', or 'MATHEESHA PATHIRANA' using:
  `LOWER(column_name) = LOWER('user_input')`
- Always ensure text comparisons use LOWER() or `COLLATE NOCASE` for correct matching.

Handle minor typos, grammar mistakes, or spelling variations to infer intent.


Always map human-readable names to internal IDs using the master tables:
- Use `Players` to map Player Name → player_id
- Use `Action` to map Action → action_id
- Use `Event` to map Event Name → event_id
- Use other master tables similarly
**Always** use a subquery instead of hardcoding or assuming IDs like `p6`.
Only use the following tables:
{table_info}
 
The following are player names and should be matched to the `Players` table (case-insensitive, with typo tolerance):
BEURAN HENDRICKS, DAVID WIESE, DONOVAN FERREIRA, DEVON CONWAY, DOUG BRACEWELL, ERIC SIMONS, EVAN JONES, FAF DU PLESSIS, GERALD COETZEE, HARDUS VILJOEN, IMRAN TAHIR, JONNY BAIRSTOW, JP KING, KASI VISWANATHAN, LAKSHMI NARAYANAN, LEUS DU PLOOY, LUTHO SIPAMLA, MAHEESH THEEKSHANA, MATHEESHA PATHIRANA, MOEEN ALI, SANJAY NATARAJAN, SIBONELO MAKHANYA, STEPHEN FLEMING, TABRAIZ SHAMSI, TOMMY SIMSEK, TSHEPO MOREKI, WIHAN LUBBE


Question: {input}
"""

ANSWER_PROMPT_TEMPLATE = """
Given the following user question, corresponding SQL query, and SQL result, respond by formatting the SQL result into a clear, structured table format based on the intent of the question.
- Do NOT hallucinate or generate data that is not present in the SQL result.
- Use relevant key names derived from the result columns and context of the question.
- Only include keys present in the SQL result.
- If the result is a list of items (e.g., images, players), present it as a table with corresponding correct column names.
- If the result is a numeric-only aggregate (e.g., COUNT, SUM, AVG), return a simple sentence answer in natural language.
- The result must have File Name, URL and Player Name when applicable.
Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer:
"""

REPHRASE_PROMPT_TEMPLATE = """
You are an assistant that rewrites follow-up questions into standalone questions by including context from the previous chat.
Use the names and entities mentioned in the previous to resolve any pronouns.
Chat History:
{chat_history}
Follow-up Question:
{question}
Standalone Question:
"""

# -------------------------------
# Utility Functions
# -------------------------------
@st.cache_data
def load_sheets():
    xl = pd.ExcelFile(EXCEL_PATH)
    return {
        name: xl.parse(name).to_dict(orient="records")
        for name in ["finalTaggedData", "Players", "Action", "Event", "Mood", "Sublocation"]
    }

@st.cache_resource
def load_into_sqlite(tables):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    for name, data in tables.items():
        pd.DataFrame(data).to_sql(name, engine, index=False, if_exists='replace')
    return engine

def clean_sql_query(text: str) -> str:
    block_pattern = r"```(?:sql|SQL)?\s*(.*?)\s*```"
    text = re.sub(block_pattern, r"\1", text, flags=re.DOTALL)
    prefix_pattern = r"^(?:SQL\s*Query|SQLQuery|MySQL|SQL|SQLite|ite)\s*:\s*"
    text = re.sub(prefix_pattern, "", text, flags=re.IGNORECASE)
    sql_statement_pattern = r"(SELECT.*?;)"
    match = re.search(sql_statement_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    return re.sub(r'\s+', ' ', text.strip())

def print_sql(x):
    print("\nLLM-generated SQL (raw):", x["raw_query"])
    print("Cleaned SQL:", x["query"])
    return x

def convert_drive_url(original_url):
    """Convert a Google Drive shareable URL to a direct image download URL."""
    if "drive.google.com" in original_url:
        if "open?id=" in original_url:
            file_id = original_url.split("open?id=")[-1]
        elif "/file/d/" in original_url:
            file_id = original_url.split("/file/d/")[1].split("/")[0]
        else:
            return original_url
        return f"https://drive.google.com/uc?id={file_id}"
    return original_url

def fetch_image_with_retry(url, retries=3):
    """Fetch image bytes with retry logic."""
    for _ in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.content
        except Exception:
            continue
    return None

def show_drive_images_from_result(result):
    try:
        start_time = time.time()
        logging.info("Starting image display process")
        
        result = result.strip()
        if not result:
            logging.warning("Empty result received")
            st.info("Empty result received. Skipping image display.")
            return

        # Handle SQL result tuples
        if isinstance(result, str) and result.startswith("[("):
            logging.info("Processing SQL result tuples")
            try:
                tuples_str = result[1:-1]
                tuples_list = []
                
                for item in tuples_str.split("),"):
                    try:
                        item = item.strip("() ")
                        if not item:
                            continue
                            
                        values = [v.strip("' ") for v in item.split(",")]
                        
                        if len(values) >= 2:
                            record = {
                                "File Name": values[0],
                                "URL": values[1]
                            }
                            if len(values) > 2:
                                record["Player Name"] = values[2]
                            else:
                                record["Player Name"] = ""
                                
                            tuples_list.append(record)
                        else:
                            logging.warning(f"Skipping invalid tuple: {item} - insufficient values")
                            
                    except Exception as e:
                        logging.error(f"Error parsing tuple {item}: {str(e)}")
                        continue
                        
                if not tuples_list:
                    logging.warning("No valid image records found in the result")
                    st.warning("No valid image records found in the result")
                    return
                    
                records = tuples_list
                
            except Exception as e:
                logging.error(f"Error parsing SQL result tuples: {str(e)}")
                st.warning(f"Error parsing SQL result tuples: {str(e)}")
                return
                
        elif isinstance(result, (list, dict)):
            records = result
        else:
            try:
                records = json.loads(result)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*\]', result)
                if json_match:
                    try:
                        records = json.loads(json_match.group(0))
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON parsing error: {str(e)}")
                        st.warning(f"Could not parse JSON from result: {e}\nResult content: {result[:200]}...")
                        return
                else:
                    logging.error("No valid JSON array found in result")
                    st.warning("No valid JSON array found in the result")
                    return

        try:
            if isinstance(records, list):
                df = pd.DataFrame(records)
            else:
                df = pd.DataFrame([records])
        except Exception as e:
            logging.error(f"DataFrame creation error: {str(e)}")
            st.warning(f"Error creating DataFrame: {str(e)}")
            return

        if {"File Name", "URL"}.issubset(df.columns):
            total_images = len(df)
            logging.info(f"Starting to load {total_images} images")
            st.markdown(f"### 📸 Loading {total_images} Images")
            
            col1, col2 = st.columns([4, 1])
            with col1:
                progress_bar = st.progress(0)
            with col2:
                stop_button = st.button("⏹️ Stop Loading")
            
            status_text = st.empty()
            
            image_load_times = []
            loaded_images = 0
            cols = st.columns(3)
            
            for idx, row in df.iterrows():
                if stop_button:
                    logging.info("Image loading stopped by user")
                    st.warning("Image loading stopped. You can continue with a new question or follow-up.")
                    break
                
                try:
                    img_start_time = time.time()
                    file_name = row.get("File Name", "Unnamed")
                    player_name = row.get("Player Name", "")
                    url = convert_drive_url(row.get("URL", ""))
                    
                    progress = (idx + 1) / total_images
                    progress_bar.progress(progress)
                    status_text.text(f"Loading image {idx + 1} of {total_images}: {file_name}")
                    
                    image_bytes = fetch_image_with_retry(url)
                    if image_bytes:
                        image = Image.open(BytesIO(image_bytes))
                        with cols[idx % 3]:
                            st.image(image, caption=f"{player_name}\n{file_name}")
                            st.markdown(f"[View Image]({url})", unsafe_allow_html=True)
                        loaded_images += 1
                        img_load_time = time.time() - img_start_time
                        image_load_times.append(img_load_time)
                        
                        logging.info(f"Successfully loaded image - File: {file_name}, Player: {player_name}, Load time: {img_load_time:.2f}s")
                except Exception as e:
                    logging.error(f"Error loading image {file_name}: {str(e)}")
                    continue
            
            total_time = time.time() - start_time
            avg_load_time = sum(image_load_times) / len(image_load_times) if image_load_times else 0
            
            logging.info(f"""
            Image Loading Session Summary:
            - Total images: {total_images}
            - Successfully loaded: {loaded_images}
            - Total processing time: {total_time:.2f}s
            - Average load time: {avg_load_time:.2f}s
            - Fastest load: {min(image_load_times):.2f}s
            - Slowest load: {max(image_load_times):.2f}s
            """)
            
            progress_bar.empty()
            status_text.empty()
            
            if stop_button:
                st.info(f"Stopped after loading {loaded_images} of {total_images} images")
            else:
                st.success(f"Successfully loaded {loaded_images} of {total_images} images")
            
        else:
            logging.error("Missing required columns in DataFrame")
            st.warning("Required columns `File Name` and `URL` not found in SQL result.")

    except Exception as e:
        error_msg = f"Unexpected error displaying images: {str(e)}\nResult type: {type(result)}"
        logging.error(error_msg)
        st.warning(error_msg)

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="SQLite Agent", layout="wide")
st.title("🧠 SQLite Q&A Agent (Gemini)")

# Initialize session state for chat history if not exists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["user"])
        else:
            st.chat_message("assistant").write(message["content"])

tables = load_sheets()
engine = load_into_sqlite(tables)
db = SQLDatabase(engine)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

table_info = db.get_table_info()
# print(f"table_info : {table_info}.......")
generate_query = create_sql_query_chain(
    llm,
    db,
    prompt=PromptTemplate(
        input_variables=["input", "table_info", "top_k"],
        template=SYSTEM_PROMPT_TEMPLATE
    ),
    k=None
)

execute_query = QuerySQLDataBaseTool(db=db)
answer_prompt = PromptTemplate.from_template(ANSWER_PROMPT_TEMPLATE)
rephrase_answer = answer_prompt | llm | StrOutputParser()

rephrase_prompt = PromptTemplate.from_template(REPHRASE_PROMPT_TEMPLATE)
rephrase_chain = (
    RunnableLambda(lambda x: {
        "chat_history": "\n".join(
            [f"User: {m['user']}" if m['role'] == "user" else f"Assistant: {m['content']}" for m in st.session_state.chat_history]
        ),
        "question": x["question"]
    }) | rephrase_prompt | llm | StrOutputParser()
)

st.markdown("#### 🧭 Is this a follow-up question?")
followup_flag = st.radio("Follow-up?", ["No", "Yes"], index=0, horizontal=True)

query_input = st.chat_input("Ask a question about the dataset...")

if query_input:
    is_followup = followup_flag == "Yes"
    final_question = query_input

    if not is_followup:
        st.session_state.chat_history = []

    if is_followup:
        try:
            final_question = rephrase_chain.invoke({
                "question": query_input,
                "messages": st.session_state.chat_history
            })
            st.markdown(f"✏️ Rephrased Question: **{final_question}**")
        except Exception as e:
            st.error(f"Error rephrasing follow-up: {e}")

    try:
        with st.spinner("🔍 Generating answer..."):
            chain_with_result_check = (
                RunnablePassthrough.assign(table_info=lambda x: table_info)
                | RunnablePassthrough.assign(
                    raw_query=generate_query,
                    query=generate_query | RunnableLambda(clean_sql_query)
                )
                | RunnableLambda(print_sql)
                | RunnablePassthrough.assign(result=itemgetter("query") | execute_query)
                | RunnableLambda(lambda x: (
                    "This information is not available in the database."
                    if not x["result"].strip()
                    else (
                        # print("🧾 SQL Result being parsed:\n", x["result"]),
                        rephrase_answer.invoke({
                            "question": x["question"],
                            "query": x["query"],
                            "result": x["result"]
                        }),
                        show_drive_images_from_result(x["result"])
                    )[-2]  # return only the LLM response
                ))
            )

            response = chain_with_result_check.invoke({
                "question": final_question,
                "messages": st.session_state.chat_history,
                "table_info": table_info
            })

            st.chat_message("user").write(query_input)
            st.chat_message("assistant").write(response)

            st.session_state.chat_history.append({"role": "user", "user": query_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
