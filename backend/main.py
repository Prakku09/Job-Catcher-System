from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List
import os

from . import models, schemas, database, scraper

# Create database tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Job Catcher API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# Dependency to get DB session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/jobs", response_model=List[schemas.Job])
def read_jobs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    jobs = db.query(models.Job).order_by(models.Job.published_at.desc()).offset(skip).limit(limit).all()
    return jobs

@app.get("/jobs/{job_id}", response_model=schemas.Job)
def read_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/jobs/fetch", response_model=List[schemas.Job])
def trigger_fetch_jobs(db: Session = Depends(get_db)):
    """
    Endpoint to trigger the scraper to fetch new jobs and store them in the database.
    """
    new_jobs = scraper.fetch_jobs_mock(db)
    return new_jobs
