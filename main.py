import os
from typing import Optional
import time

from fastapi import FastAPI, HTTPException,Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from agent.agent import complete


class ResponseTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        end_time = time.perf_counter()
        process_time = end_time - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response




app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(ResponseTimeMiddleware)

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


@app.get("/complete")
def get_completion(prompt: str):
    print(prompt)
    return {"response": complete(prompt)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8002)),
        reload=True,
    )
