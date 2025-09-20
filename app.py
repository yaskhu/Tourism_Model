import os
import pandas as pd
import pickle
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# 🔑 Set Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA7wUkLMdT097N4JkslXZL51_tae1k66dw"

# 📂 Load dataset
CSV_PATH = r"C:\\Sheets\\SIH\\ChatBot\\Jharkhand_Journeys - Sheet1.csv"
df = pd.read_csv(CSV_PATH, encoding="latin1")

# 📑 Prepare documents
def make_doc(row):
    parts = [
        "Place: " + str(row['Place Name']),
        ("Best time to visit: " + str(row['Best time to visit'])) if pd.notna(row['Best time to visit']) else "",
        ("Required days: " + str(row['Number of days to complete the visit'])) if pd.notna(row['Number of days to complete the visit']) else "",
        ("Entry fee: " + str(row['Entry fee'])) if pd.notna(row['Entry fee']) else "",
        ("Coordinates: " + str(row['Coordinates'])) if pd.notna(row['Coordinates']) else "",
        ("Best Food: " + str(row['Best Food'])) if pd.notna(row['Best Food']) else "",
        ("Culture: " + str(row['Key culture of that place'])) if pd.notna(row['Key culture of that place']) else "",
        ("Neighbouring Districts: " + str(row['Neighbouring Districts'])) if pd.notna(row['Neighbouring Districts']) else "",
        ("Timings: " + str(row['Opening And Closing Timings'])) if pd.notna(row['Opening And Closing Timings']) else "",
        ("Nearby Handicraft: " + str(row['NearBy Handicraft'])) if pd.notna(row['NearBy Handicraft']) else "",
        ("Nearby Handicraft Coordinates: " + str(row['Nearby handicraft coordinates'])) if pd.notna(row['Nearby handicraft coordinates']) else "",
        ("Category: " + str(row['Category of that Place'])) if pd.notna(row['Category of that Place']) else ""
    ]
    return " | ".join([p for p in parts if p])

docs = df.apply(make_doc, axis=1).tolist()

# 📦 Vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        vectorstore = pickle.load(f)
else:
    vectorstore = FAISS.from_texts(docs, embeddings)
    with open("model.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

# 🤖 Gemini LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# 🔹 Prompt template
prompt_template = """
Answer the following user query as clearly and directly as possible.

Context (you may use it if relevant):
{context}

Question: {question}

Rules:
- Do not say "based on the provided data" or similar phrases.
- If the dataset has the answer → use it directly.
- If not, answer naturally from your own knowledge.
- Keep the answer concise and user-friendly.
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 💾 Memory with explicit output_key
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

# 🔹 QA Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    return_source_documents=True,
    output_key="answer"
)

# 🚀 FastAPI setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Query(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(query: Query):
    try:
        loop = asyncio.get_event_loop()
        # Run synchronous qa_chain safely in a thread
        result = await loop.run_in_executor(None, qa_chain, {"question": query.question})
        return {"answer": result["answer"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🏃 Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)