const API_BASE_URL = 'http://localhost:8000';
let jobsData = [];

const fetchBtn = document.getElementById('fetch-btn');
const searchInput = document.getElementById('search-input');
const jobsContainer = document.getElementById('jobs-container');
const loadingIndicator = document.getElementById('loading');

async function loadJobs() {
    try {
        const response = await fetch(`${API_BASE_URL}/jobs`);
        if (!response.ok) throw new Error('Failed to fetch jobs');
        jobsData = await response.json();
        renderJobs(jobsData);
    } catch (error) {
        console.error('Error loading jobs:', error);
        jobsContainer.innerHTML = '<p style="color: var(--danger-color); grid-column: 1/-1; text-align: center;">Failed to connect to the server. Please ensure the backend is running.</p>';
    }
}

async function catchNewJobs() {
    const originalText = fetchBtn.innerText;
    fetchBtn.innerText = 'Catching...';
    fetchBtn.disabled = true;
    loadingIndicator.classList.remove('hidden');
    jobsContainer.innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE_URL}/jobs/fetch`, { method: 'POST' });
        if (!response.ok) throw new Error('Failed to fetch new jobs');
        const newJobs = await response.json();
        
        // Show success briefly
        fetchBtn.innerText = `Caught ${newJobs.length} Jobs!`;
        
        setTimeout(() => {
            fetchBtn.innerText = originalText;
            fetchBtn.disabled = false;
        }, 2000);
        
        await loadJobs();
    } catch (error) {
        console.error('Error catching jobs:', error);
        fetchBtn.innerText = originalText;
        fetchBtn.disabled = false;
    } finally {
        loadingIndicator.classList.add('hidden');
    }
}

function renderJobs(jobs) {
    jobsContainer.innerHTML = '';
    
    if (jobs.length === 0) {
        jobsContainer.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: var(--text-secondary);">No jobs found. Click "Catch New Jobs" to get started!</p>';
        return;
    }
    
    jobs.forEach((job, index) => {
        const date = new Date(job.published_at).toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
        
        const card = document.createElement('div');
        card.className = 'job-card';
        card.style.animationDelay = `${index * 0.05}s`;
        
        card.innerHTML = `
            <div class="job-header">
                <div>
                    <h3 class="job-title">${job.title}</h3>
                    <div class="job-company">${job.company}</div>
                </div>
            </div>
            <div class="job-meta">
                <span>📍 ${job.location}</span>
                <span>🏢 ${job.source}</span>
            </div>
            <p class="job-description">${job.description}</p>
            <div class="job-footer">
                <span class="job-date">${date}</span>
                <a href="${job.url}" target="_blank" class="btn-apply">View & Apply</a>
            </div>
        `;
        
        jobsContainer.appendChild(card);
    });
}

function filterJobs(query) {
    const lowerQuery = query.toLowerCase();
    const filtered = jobsData.filter(job => 
        job.title.toLowerCase().includes(lowerQuery) ||
        job.company.toLowerCase().includes(lowerQuery) ||
        job.location.toLowerCase().includes(lowerQuery)
    );
    renderJobs(filtered);
}

// Event Listeners
fetchBtn.addEventListener('click', catchNewJobs);
searchInput.addEventListener('input', (e) => filterJobs(e.target.value));

// Initial load
loadJobs();
