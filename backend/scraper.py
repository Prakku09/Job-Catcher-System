import requests
from sqlalchemy.orm import Session
from . import models, schemas
import uuid
import random
from datetime import datetime

def fetch_jobs_mock(db: Session):
    """
    Mock scraper that generates dummy job data.
    In a real scenario, this would use BeautifulSoup or requests to hit an API/Website.
    """
    titles = ["Frontend Developer", "Backend Engineer", "Data Scientist", "Product Manager", "DevOps Engineer", "UI/UX Designer"]
    companies = ["TechCorp", "Innovate LLC", "Global Solutions", "Startup Inc.", "Big Data Co.", "Creative Agency"]
    locations = ["Remote", "New York, NY", "San Francisco, CA", "London, UK", "Berlin, DE", "Austin, TX"]
    
    new_jobs_count = random.randint(3, 8)
    added_jobs = []
    
    for _ in range(new_jobs_count):
        job_data = schemas.JobCreate(
            title=random.choice(titles),
            company=random.choice(companies),
            location=random.choice(locations),
            description="This is an exciting opportunity to work with a dynamic team using cutting-edge technologies. You will be responsible for designing and developing scalable solutions.",
            url=f"https://example.com/jobs/{uuid.uuid4()}",
            source="Mock Scraper",
            published_at=datetime.utcnow()
        )
        
        # Check if URL exists to avoid duplicates
        existing_job = db.query(models.Job).filter(models.Job.url == job_data.url).first()
        if not existing_job:
            db_job = models.Job(**job_data.dict())
            db.add(db_job)
            db.commit()
            db.refresh(db_job)
            added_jobs.append(db_job)
            
    return added_jobs
