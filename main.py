from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from groq import Groq

# NEW: Import for environment variables
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="ITMS AI Customer Care")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NEW: Read the API key from the .env file
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- MODELS ---
class ChatMessage(BaseModel):
    message: str
    user_name: str = "User"
    role: str = "Driver"

class ReportRequest(BaseModel):
    total_vehicles: int
    total_revenue: float
    mpesa_revenue: float
    cash_revenue: float
    peak_hour: str

class DriverInsightRequest(BaseModel):
    driver_name: str
    total_spent: float
    total_payments: int
    favorite_method: str

class DisputeSummaryRequest(BaseModel):
    driver_name: str
    complaint: str

# --- ROUTES ---
@app.get("/")
def read_root():
    return {"message": "ITMS AI Brain is Online!"}

@app.post("/api/chat")
async def chat_with_ai(chat: ChatMessage):
    print(f"💬 Message from {chat.user_name} ({chat.role}): {chat.message}")
    
    system_prompt = f"""You are the official AI Customer Care Assistant for the Intelligent Toll Management System (ITMS) in Kenya. 
You are currently speaking to a {chat.role} named {chat.user_name}.

Your job is to help them based on their role:
- If they are a DRIVER: Help with M-Pesa payments, toll fees (Cars: 200 KES, Trucks: 500 KES), and vehicle registration.
- If they are an ADMIN: Help with system reports, managing users, viewing revenue, and toll gate settings.
- If they are a TOLL ATTENDANT: Help with processing cash, verifying vehicles, and managing their shift.

RULES:
- Keep answers short, polite, and professional.
- If the user asks a question NOT related to the toll system (e.g., general knowledge, coding, politics), reply exactly with: "I am sorry, but I am only programmed to assist with Intelligent Toll Management System queries."
"""

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat.message}
            ],
            temperature=0.3,
            max_tokens=250
        )
        
        reply = completion.choices[0].message.content
        print(f"🤖 AI Reply: {reply}")
        
        return {"success": True, "reply": reply}
        
    except Exception as e:
        print(f"Error: {e}")
        return {"success": False, "reply": "Sorry, my AI brain is currently offline."}

@app.post("/api/generate-report")
async def generate_report(data: ReportRequest):
    print("📊 Generating daily report...")
    
    prompt = f"""
    You are the AI Data Analyst for the ITMS. Write a professional, 3-sentence summary of today's performance for the Admin.
    Here is the data:
    - Total Vehicles Processed: {data.total_vehicles}
    - Total Revenue: KES {data.total_revenue}
    - M-Pesa Revenue: KES {data.mpesa_revenue}
    - Cash Revenue: KES {data.cash_revenue}
    - Peak Hour: {data.peak_hour}

    Make it sound like a business intelligence report. Do not use markdown formatting, just plain text.
    """

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=250
        )
        report = completion.choices[0].message.content
        print(f"📝 Report generated: {report}")
        return {"success": True, "report": report}
    except Exception as e:
        print(f"Report Error: {e}")
        return {"success": False, "report": "Could not generate report."}

@app.post("/api/driver-insight")
async def driver_insight(data: DriverInsightRequest):
    print(f"💡 Generating insight for driver: {data.driver_name}")
    
    prompt = f"""
    You are the AI assistant for the ITMS. The driver {data.driver_name} has:
    - Spent a total of KES {data.total_spent} on tolls.
    - Made {data.total_payments} payments.
    - Their most used payment method is {data.favorite_method}.
    
    Write a short, friendly, 2-sentence tip for the driver. 
    If they use M-Pesa a lot, suggest they use the ITMS Wallet to save time. 
    Keep it conversational, starting with 'Hey {data.driver_name}!' and do not use markdown.
    """

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        insight = completion.choices[0].message.content
        print(f"💡 Insight generated: {insight}")
        return {"success": True, "insight": insight}
    except Exception as e:
        print(f"Insight Error: {e}")
        return {"success": False, "insight": "Keep driving safely!"}

@app.post("/api/summarize-dispute")
async def summarize_dispute(data: DisputeSummaryRequest):
    print(f"⚖️ Summarizing dispute from {data.driver_name}")
    
    prompt = f"""
    You are the AI Support Assistant for the ITMS. A driver named {data.driver_name} has submitted the following complaint:
    "{data.complaint}"

    Write a short, 1-sentence summary of this complaint for the Admin to read. 
    Start with "Summary:". Do not use markdown. Be direct.
    """
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        summary = completion.choices[0].message.content
        print(f"⚖️ Summary generated: {summary}")
        return {"success": True, "summary": summary}
    except Exception as e:
        print(f"Dispute Error: {e}")
        return {"success": False, "summary": "Could not summarize dispute."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)