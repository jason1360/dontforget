import sqlite3
import datetime
import os
import json
import re
from typing import List
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- CONFIGURATION ---
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SERVER_SECRET = os.getenv("DONTFORGET_SECRET_KEY")
DB_PATH = "dontforget.db"
MODEL_ID = "gemini-2.0-flash" 

if not GEMINI_KEY or not SERVER_SECRET:
    raise ValueError("Missing keys in .env file.")

app = FastAPI(title="DontForget - God Mode")
client = genai.Client(api_key=GEMINI_KEY)

# --- SECURITY ---
async def verify_api_key(x_api_key: str = Header(..., description="Server Secret")):
    if x_api_key != SERVER_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key

# --- DATA MODELS ---
class ThoughtRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    question: str

# --- DATABASE LAYER ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    # UNINDEXED timestamp saves space
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory 
        USING fts5(text, tags, intent, timestamp, UNINDEXED);
    """)
    conn.commit()
    conn.close()

init_db()

# --- TOOLS ---
def execute_sql(sql_query: str):
    """
    TOOL: Executes a read-only SQL query.
    Useful for SEARCHING. Returns rows with 'rowid'.
    """
    try:
        # Prevent deletion via SQL injection in the search tool
        if any(x in sql_query.upper() for x in ["DELETE", "DROP", "UPDATE", "INSERT"]):
            return "Error: Use the delete_memories tool for deletion."
        
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row 
        # Select rowid explicitly so we can delete later
        cursor = conn.execute(sql_query)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if not rows: return "No records found."
        return json.dumps(rows, default=str)
    except Exception as e:
        return f"SQL Error: {e}"

def delete_memories(rowids: List[int]):
    """
    TOOL: Deletes memories by their Row IDs.
    Use this ONLY when the user explicitly asks to forget/delete something.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        # Flatten list if needed or handle single int
        if isinstance(rowids, int): rowids = [rowids]
        
        placeholders = ','.join('?' * len(rowids))
        sql = f"DELETE FROM memory WHERE rowid IN ({placeholders})"
        cursor = conn.execute(sql, rowids)
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        return f"Successfully deleted {deleted_count} memories."
    except Exception as e:
        return f"Deletion Error: {e}"

# --- ENDPOINTS ---
@app.post("/remember", dependencies=[Depends(verify_api_key)])
def remember(request: ThoughtRequest):
    try:
        prompt = f"""
        Analyze input: "{request.text}"
        Return JSON:
        - "tags": csv keywords.
        - "intent": ONE word [task, fact, idea, list].
        """
        resp = client.models.generate_content(
            model=MODEL_ID, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(resp.text)
        tags = data.get("tags", "general")
        intent = data.get("intent", "fact")
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO memory (text, tags, intent, timestamp) VALUES (?, ?, ?, ?)", 
                         (request.text, tags, intent, ts))
        return {"status": "saved", "tags": tags}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/remind", dependencies=[Depends(verify_api_key)])
def remind(request: QueryRequest):
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        sys_instruct = f"""
        You are 'DontForget'. Date: {today}.
        
        TOOLS:
        1. `execute_sql(query)`: Search for info. Always select 'rowid'.
        2. `delete_memories(rowids)`: Delete specific rows.

        PROTOCOL:
        - **SEARCH**: `SELECT rowid, text, intent, timestamp FROM memory...`
        - **DELETE**: If user says "Delete X", FIRST search for X to get the 'rowid', THEN call `delete_memories([id])`.
        - **CONFIRM**: If multiple matches found for deletion, ask user for clarification (unless they said "delete all").
        - **ZERO-HESITATION**: For retrieval, just run the SQL.
        """
        
        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                tools=[execute_sql, delete_memories], 
                system_instruction=sys_instruct,
                temperature=0.1
            )
        )
        
        response = chat.send_message(request.question)
        
        # --- ROBUST LOOP ---
        turn = 0
        while turn < 5:
            turn += 1
            if not response.candidates or not response.candidates[0].content.parts:
                break
            
            part = response.candidates[0].content.parts[0]
            
            # Handle Tool Call
            if part.function_call:
                fname = part.function_call.name
                args = part.function_call.args
                print(f"🤖 Tool: {fname} | Args: {args}")
                
                if fname == "execute_sql":
                    result = execute_sql(args["sql_query"])
                elif fname == "delete_memories":
                    # Handle list argument safely
                    ids = args["rowids"]
                    if isinstance(ids, float): ids = [int(ids)] # JSON fix
                    result = delete_memories(ids)
                
                response = chat.send_message(
                    types.Part(function_response=types.FunctionResponse(
                        name=fname, response={"result": result}
                    ))
                )
                continue
            
            # Handle Lazy Text SQL
            text = part.text or ""
            if "```sql" in text:
                match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL)
                if match:
                    result = execute_sql(match.group(1))
                    response = chat.send_message(f"SYSTEM: Executed SQL. Result: {result}")
                    continue
            
            break
            
        return {"answer": response.text}

    except Exception as e:
        print(e)
        raise HTTPException(500, "Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
