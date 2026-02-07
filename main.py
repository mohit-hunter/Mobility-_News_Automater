from fastapi import FastAPI, HTTPException, Query, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from services import ScraperService
import os
import uvicorn

# Import beta modules conditionally
from config import BETA_MODE, CATEGORIES

app = FastAPI(title="Frost News Collector")

# Initialize Service
service = ScraperService()

# Beta imports (only if BETA_MODE is enabled)
if BETA_MODE:
    from beta_auth import (
        get_user_by_email, 
        generate_token, 
        validate_token, 
        invalidate_token,
        update_last_login,
        can_review
    )
    from beta_classifier import classify_news, save_correction, generate_news_id

# Models
class FeedItem(BaseModel):
    name: str
    url: str

class LoginRequest(BaseModel):
    email: str

class CorrectionRequest(BaseModel):
    news_id: str
    original_prediction: str
    corrected_label: str


# Helper to get current user from token
def get_current_user(authorization: Optional[str] = None):
    """Extract and validate user from Authorization header."""
    if not BETA_MODE:
        return None
    
    if not authorization:
        return None
    
    # Expect "Bearer <token>"
    parts = authorization.split(' ')
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    
    token = parts[1]
    return validate_token(token)


# ============================================
# BETA AUTH ENDPOINTS (only if BETA_MODE)
# ============================================

if BETA_MODE:
    @app.post("/api/auth/login")
    async def login(request: LoginRequest):
        """Login with email (beta - no password required)."""
        user = get_user_by_email(request.email)
        if not user:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        
        token = generate_token(user['user_id'])
        update_last_login(user['user_id'])
        
        return {
            "token": token,
            "user": {
                "user_id": user['user_id'],
                "email": user['email'],
                "role": user['role'],
                "can_review": can_review(user)
            }
        }
    
    @app.post("/api/auth/logout")
    async def logout(authorization: Optional[str] = Header(None)):
        """Logout and invalidate token."""
        if authorization:
            parts = authorization.split(' ')
            if len(parts) == 2:
                invalidate_token(parts[1])
        return {"message": "Logged out"}
    
    @app.get("/api/auth/me")
    async def get_me(authorization: Optional[str] = Header(None)):
        """Get current user info."""
        user = get_current_user(authorization)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        return {
            "user_id": user['user_id'],
            "email": user['email'],
            "role": user['role'],
            "can_review": can_review(user)
        }
    
    @app.post("/api/corrections")
    async def submit_correction(
        correction: CorrectionRequest,
        authorization: Optional[str] = Header(None)
    ):
        """Submit a label correction (Admin/Reviewer only)."""
        user = get_current_user(authorization)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        if not can_review(user):
            raise HTTPException(status_code=403, detail="Permission denied: reviewers only")
        
        if correction.corrected_label not in CATEGORIES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid category. Must be one of: {CATEGORIES}"
            )
        
        try:
            result = save_correction(
                news_id=correction.news_id,
                original_prediction=correction.original_prediction,
                corrected_label=correction.corrected_label,
                reviewer_id=user['user_id']
            )
            return {"message": "Correction saved", "correction": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/categories")
    async def get_categories():
        """Get available classification categories."""
        return {"categories": CATEGORIES}


# ============================================
# EXISTING API ENDPOINTS
# ============================================

@app.get("/api/news")
async def get_news(
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"), 
    end_date: str = Query(None, description="End date (YYYY-MM-DD)"),
    search: str = Query(None, description="Search query")
):
    """Get news items with optional filtering."""
    try:
        data = service.get_news(start_date, end_date, search)
        
        # Add classification if BETA_MODE is enabled
        if BETA_MODE:
            data = classify_news(data)
        
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scrape")
async def trigger_scrape():
    """Trigger the scraper manually."""
    try:
        result = service.run_scraper()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feeds")
async def get_feeds():
    """List all configured RSS feeds."""
    return service.get_feeds()

@app.post("/api/feeds")
async def add_feed(feed: FeedItem):
    """Add a new RSS feed."""
    success, message = service.add_feed(feed.name, feed.url)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}

@app.delete("/api/feeds/{name}")
async def remove_feed(name: str):
    """Remove an RSS feed."""
    success, message = service.remove_feed(name)
    if not success:
        raise HTTPException(status_code=404, detail=message)
    return {"message": message}

# ============================================
# STATIC FILES & FRONTEND
# ============================================

# Ensure static directory exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.get("/login")
async def login_page():
    """Serve login page (beta only)."""
    if BETA_MODE:
        return FileResponse('static/login.html')
    return FileResponse('static/index.html')

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
