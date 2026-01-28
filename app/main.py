from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import AskReq, AskResponse
from app.agent import Agent
from app.guardrails import SecurityError
import logging

logging.basicConfig(filename='app.log', level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Analyst Agent")
agent = Agent()

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskReq):
    try:
        response = agent.run(request.query)
        return response
    except SecurityError as e:
        logger.warning(f"SecurityError: {e}")
        return JSONResponse(status_code=403, content={"detail": str(e), "error_type": "SecurityError"})
    except TimeoutError as e:
        logger.error(f"TimeoutError: {e}")
        return JSONResponse(status_code=408, content={"detail": str(e), "error_type": "TimeoutError"})
    except ValueError as e: 
        logger.warning(f"ValueError: {e}")
        return JSONResponse(status_code=400, content={"detail": str(e), "error_type": "ValidationError"})
    except Exception as e:
        logger.exception("Unexpected error")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "error_type": "InternalError"})

@app.get("/health")
def health():
    return {"status": "ok"}
