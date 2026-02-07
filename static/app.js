const API_BASE = '/api';

// Auth State
const auth = {
    token: localStorage.getItem('authToken'),
    user: JSON.parse(localStorage.getItem('currentUser') || 'null')
};

// State
const state = {
    news: [],
    feeds: [],
    loading: false,
    categories: ['automakers', 'government', 'suppliers']
};

// DOM Elements
const newsGrid = document.getElementById('news-grid');
const runScraperBtn = document.getElementById('run-scraper-btn');
const manageFeedsBtn = document.getElementById('manage-feeds-btn');
const feedModal = document.getElementById('feed-modal');
const closeModalBtn = document.querySelector('.close-modal');
const feedList = document.getElementById('feed-list');
const addFeedBtn = document.getElementById('add-feed-btn');
const statusBar = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');
const themeToggle = document.getElementById('theme-toggle');
const startDateInput = document.getElementById('start-date');
const endDateInput = document.getElementById('end-date');
const searchInput = document.getElementById('search-query');
const searchBtn = document.getElementById('search-btn');
const downloadCsvBtn = document.getElementById('download-csv-btn');

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    loadNews();
    setupTheme();
    setupEventListeners();
    setupUserDisplay();
});

// Auth Functions
function checkAuth() {
    // If no token, redirect to login
    if (!auth.token) {
        window.location.href = '/login';
        return;
    }

    // Verify token is still valid
    fetch(`${API_BASE}/auth/me`, {
        headers: { 'Authorization': `Bearer ${auth.token}` }
    })
        .then(res => {
            if (!res.ok) {
                logout();
            }
        })
        .catch(() => logout());
}

function logout() {
    fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${auth.token}` }
    }).catch(() => { });

    localStorage.removeItem('authToken');
    localStorage.removeItem('currentUser');
    window.location.href = '/login';
}

function setupUserDisplay() {
    const header = document.querySelector('.glass-header');
    if (!header || !auth.user) return;

    // Check if user-info already exists
    if (document.getElementById('user-info')) return;

    // Create user info element
    const userInfo = document.createElement('div');
    userInfo.id = 'user-info';
    userInfo.className = 'user-info';
    userInfo.innerHTML = `
        <span class="user-email">${auth.user.email}</span>
        <span class="user-role ${auth.user.role}">${auth.user.role}</span>
        <button id="logout-btn" class="icon-btn" title="Logout">
            <i class="fa-solid fa-right-from-bracket"></i>
        </button>
    `;

    // Insert before theme toggle
    header.insertBefore(userInfo, themeToggle);

    // Add logout handler
    document.getElementById('logout-btn').addEventListener('click', logout);
}

function setupEventListeners() {
    runScraperBtn.addEventListener('click', runScraper);
    manageFeedsBtn.addEventListener('click', openFeedModal);
    closeModalBtn.addEventListener('click', () => feedModal.classList.add('hidden'));
    addFeedBtn.addEventListener('click', addFeed);
    downloadCsvBtn.addEventListener('click', downloadCsv);
    themeToggle.addEventListener('click', toggleTheme);

    // Filters
    startDateInput.addEventListener('change', loadNews);
    endDateInput.addEventListener('change', loadNews);
    searchBtn.addEventListener('click', loadNews);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') loadNews();
    });

    // Close modal on outside click
    feedModal.addEventListener('click', (e) => {
        if (e.target === feedModal) feedModal.classList.add('hidden');
    });
}

function setupTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('i');
    if (theme === 'dark') {
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    } else {
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
    }
}

// API Calls
async function loadNews() {
    setLoading(true, "Loading news...");
    try {
        const params = new URLSearchParams();
        if (startDateInput.value) params.append('start_date', startDateInput.value);
        if (endDateInput.value) params.append('end_date', endDateInput.value);
        if (searchInput.value) params.append('search', searchInput.value);

        const response = await fetch(`${API_BASE}/news?${params}`);
        if (!response.ok) throw new Error('Failed to fetch news');

        state.news = await response.json();
        renderNews();
    } catch (error) {
        console.error(error);
        showStatus(error.message, 'error');
    } finally {
        setLoading(false);
    }
}

async function runScraper() {
    setLoading(true, "Running scraper... This may take a while.");
    try {
        const response = await fetch(`${API_BASE}/scrape`, { method: 'POST' });
        if (!response.ok) throw new Error('Scraper failed');

        const result = await response.json();
        showStatus(`Scrape complete! New articles: ${result.new_articles}`, 'success');
        loadNews(); // Reload news to show new ones
    } catch (error) {
        console.error(error);
        showStatus(error.message, 'error');
        setLoading(false);
    }
}

async function loadFeeds() {
    try {
        const response = await fetch(`${API_BASE}/feeds`);
        state.feeds = await response.json();
        renderFeeds();
    } catch (error) {
        console.error(error);
    }
}

async function addFeed() {
    const name = document.getElementById('new-feed-name').value;
    const url = document.getElementById('new-feed-url').value;

    if (!name || !url) return alert("Please fill in both fields");

    try {
        const response = await fetch(`${API_BASE}/feeds`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, url })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail);
        }

        document.getElementById('new-feed-name').value = '';
        document.getElementById('new-feed-url').value = '';
        loadFeeds();
    } catch (error) {
        alert(error.message);
    }
}

async function removeFeed(name) {
    if (!confirm(`Are you sure you want to delete ${name}?`)) return;

    try {
        const response = await fetch(`${API_BASE}/feeds/${encodeURIComponent(name)}`, {
            method: 'DELETE'
        });

        if (!response.ok) throw new Error('Failed to delete feed');
        loadFeeds();
    } catch (error) {
        alert(error.message);
    }
}

async function submitCorrection(newsId, originalCategory, newCategory) {
    try {
        const response = await fetch(`${API_BASE}/corrections`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.token}`
            },
            body: JSON.stringify({
                news_id: newsId,
                original_prediction: originalCategory,
                corrected_label: newCategory
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail);
        }

        showStatus('Label correction saved!', 'success');
        loadNews(); // Reload to show updated label
    } catch (error) {
        showStatus(error.message, 'error');
    }
}

// Rendering
function renderNews() {
    newsGrid.innerHTML = '';

    if (state.news.length === 0) {
        newsGrid.innerHTML = '<div class="empty-state">No news found for the selected criteria.</div>';
        return;
    }

    const canReview = auth.user && auth.user.can_review;

    state.news.forEach(item => {
        const card = document.createElement('article');
        card.className = 'news-card';

        // Category badge color
        const categoryClass = item.predicted_category || 'unknown';
        const confidence = item.prediction_confidence || 0;
        const confidencePercent = Math.round(confidence * 100);
        const isCorrection = item.is_corrected;

        // Build category badge HTML
        let categoryBadge = '';
        if (item.predicted_category) {
            categoryBadge = `
                <div class="category-badge ${categoryClass} ${isCorrection ? 'corrected' : ''}">
                    <span class="category-label">${item.predicted_category}</span>
                    <span class="category-confidence">${confidencePercent}%</span>
                </div>
            `;
        }

        // Build correction UI for reviewers
        let correctionUI = '';
        if (canReview && item.news_id) {
            correctionUI = `
                <div class="correction-controls">
                    <select class="correction-select" data-news-id="${item.news_id}" data-original="${item.predicted_category || ''}">
                        <option value="">Correct label...</option>
                        ${state.categories.map(cat =>
                `<option value="${cat}" ${cat === item.predicted_category ? 'disabled' : ''}>${cat}</option>`
            ).join('')}
                    </select>
                </div>
            `;
        }

        card.innerHTML = `
            <div class="news-meta">
                <span class="news-source">${item.Source}</span>
                <span class="news-date">${item.Date}</span>
            </div>
            ${categoryBadge}
            <h3 class="news-title">${item['Executive Headline']}</h3>
            <p class="news-summary">${item.Summary}</p>
            <div class="news-footer">
                <a href="${item['Source URL']}" target="_blank" class="news-link">
                    Read Source <i class="fa-solid fa-external-link-alt"></i>
                </a>
                ${correctionUI}
            </div>
        `;

        newsGrid.appendChild(card);
    });

    // Add event listeners for correction selects
    if (canReview) {
        document.querySelectorAll('.correction-select').forEach(select => {
            select.addEventListener('change', (e) => {
                const newsId = e.target.dataset.newsId;
                const original = e.target.dataset.original;
                const newCategory = e.target.value;

                if (newCategory && confirm(`Change category from "${original}" to "${newCategory}"?`)) {
                    submitCorrection(newsId, original, newCategory);
                } else {
                    e.target.value = ''; // Reset selection
                }
            });
        });
    }
}

function renderFeeds() {
    feedList.innerHTML = '';
    state.feeds.forEach(feed => {
        const li = document.createElement('li');
        li.className = 'feed-item';
        li.innerHTML = `
            <span>${feed.name}</span>
            <button class="delete-feed" onclick="removeFeed('${feed.name}')">
                <i class="fa-solid fa-trash"></i>
            </button>
        `;
        feedList.appendChild(li);
    });
}

function openFeedModal() {
    feedModal.classList.remove('hidden');
    loadFeeds();
}

function downloadCsv() {
    if (state.news.length === 0) {
        return showStatus('No data to download', 'error');
    }

    // Include classification data in CSV
    const headers = ['Cluster ID', 'Date', 'Executive Headline', 'Summary', 'Source URL', 'Source', 'Category', 'Confidence', 'execution_timestamp'];

    const csvRows = [
        headers.join(',')
    ];

    state.news.forEach(item => {
        const row = headers.map(header => {
            let val;
            if (header === 'Category') {
                val = item.predicted_category || '';
            } else if (header === 'Confidence') {
                val = item.prediction_confidence || '';
            } else {
                val = item[header] !== undefined && item[header] !== null ? item[header] : '';
            }
            // Escape quotes and wrap in quotes
            const escaped = ('' + val).replace(/"/g, '""');
            return `"${escaped}"`;
        });
        csvRows.push(row.join(','));
    });

    const csvContent = csvRows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', `news_export_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Utilities
function setLoading(isLoading, message = '') {
    state.loading = isLoading;
    if (isLoading) {
        statusBar.classList.remove('hidden');
        statusText.textContent = message;
        runScraperBtn.disabled = true;
    } else {
        statusBar.classList.add('hidden');
        runScraperBtn.disabled = false;
    }
}

function showStatus(message, type = 'info') {
    statusBar.classList.remove('hidden');
    statusText.textContent = message;

    // Auto hide after 3 seconds if not loading
    setTimeout(() => {
        if (!state.loading) statusBar.classList.add('hidden');
    }, 5000);
}

// Global expose for onclick events
window.removeFeed = removeFeed;
