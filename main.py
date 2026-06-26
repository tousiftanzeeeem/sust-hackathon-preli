import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # ── Serve React frontend (no build step — React loaded from CDN) ──
# # On Vercel the static folder is not available – the frontend lives on Netlify.
# if not os.getenv("VERCEL") and os.path.isdir("static"):
#     app.mount("/static", StaticFiles(directory="static"), name="static")

#     @app.get("/")
#     def serve_static():
#         return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/sort-ticket", response_model=TicketResponse)
def sort_ticket(ticket_data: TicketRequest):
    if not ticket_data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")



    try:
        result: TicketAnalysisResult = analyze_ticket(
            ticket_id=ticket_data.ticket_id,
            message=ticket_data.message,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return TicketResponse(
        ticket_id=result.ticket_id,
        case_type=result.case_type,
        severity=result.severity,
        department=result.department,
        agent_summary=result.agent_summary,
        human_review_required=result.human_review_required,
        confidence=result.confidence,
    )




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )
