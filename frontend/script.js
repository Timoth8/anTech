// Configuration
const API_URL = 'http://localhost:8000';  // Change this to your deployed backend URL

// Example texts
const examples = [
    "Pemerintah Indonesia mengumumkan kebijakan baru terkait pengembangan energi terbarukan. Menteri Energi dan Sumber Daya Mineral menyatakan bahwa target penggunaan energi terbarukan akan ditingkatkan menjadi 23% pada tahun 2025. Program ini mencakup pembangunan pembangkit listrik tenaga surya dan angin di berbagai daerah.",
    "HEBOH! Presiden tertangkap kamera sedang bertemu dengan alien di Istana Negara! Foto bocoran menunjukkan makhluk asing yang diduga berasal dari planet Mars. Saksi mata mengklaim telah melihat UFO mendarat di halaman istana tengah malam. Pemerintah diduga menyembunyikan fakta ini dari publik!"
];

// Analyze news function
async function analyzeNews() {
    const textArea = document.getElementById('newsText');
    const text = textArea.value.trim();
    
    // Validation
    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }
    
    if (text.length < 10) {
        showError('Text is too short. Please enter at least 10 characters.');
        return;
    }
    
    // Hide previous results/errors
    hideResults();
    hideError();
    
    // Show loading state
    setLoading(true);
    
    const startTime = Date.now();
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const data = await response.json();
        const processTime = Date.now() - startTime;
        
        displayResults(data, processTime);
        
    } catch (error) {
        console.error('Error:', error);
        showError(`Failed to analyze: ${error.message}. Make sure the backend server is running on ${API_URL}`);
    } finally {
        setLoading(false);
    }
}

// Display results
function displayResults(data, processTime) {
    const resultSection = document.getElementById('resultSection');
    const predictionCard = document.getElementById('predictionCard');
    const predictionText = document.getElementById('predictionText');
    const confidenceValue = document.getElementById('confidenceValue');
    const realProb = document.getElementById('realProb');
    const fakeProb = document.getElementById('fakeProb');
    const realBar = document.getElementById('realBar');
    const fakeBar = document.getElementById('fakeBar');
    const textLength = document.getElementById('textLength');
    const processTimeElement = document.getElementById('processTime');
    
    // Update prediction
    predictionText.textContent = data.prediction;
    confidenceValue.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    
    // Update card style
    predictionCard.className = 'prediction-card';
    if (data.prediction === 'REAL') {
        predictionCard.classList.add('real');
    } else {
        predictionCard.classList.add('fake');
    }
    
    // Update probabilities
    const realPercent = (data.probabilities.real * 100).toFixed(1);
    const fakePercent = (data.probabilities.fake * 100).toFixed(1);
    
    realProb.textContent = `${realPercent}%`;
    fakeProb.textContent = `${fakePercent}%`;
    
    // Animate bars
    setTimeout(() => {
        realBar.style.width = `${realPercent}%`;
        fakeBar.style.width = `${fakePercent}%`;
    }, 100);
    
    // Update stats
    textLength.textContent = data.text_length;
    processTimeElement.textContent = processTime;
    
    // Show results
    resultSection.classList.remove('hidden');
    
    // Smooth scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Loading state
function setLoading(isLoading) {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const loader = document.getElementById('loader');
    
    if (isLoading) {
        analyzeBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
    } else {
        analyzeBtn.disabled = false;
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
    }
}

// Show error
function showError(message) {
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
    
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Hide error
function hideError() {
    const errorSection = document.getElementById('errorSection');
    errorSection.classList.add('hidden');
}

// Hide results
function hideResults() {
    const resultSection = document.getElementById('resultSection');
    resultSection.classList.add('hidden');
    
    // Reset bars
    document.getElementById('realBar').style.width = '0%';
    document.getElementById('fakeBar').style.width = '0%';
}

// Clear text
function clearText() {
    document.getElementById('newsText').value = '';
    hideResults();
    hideError();
}

// Load example
function loadExample(index) {
    document.getElementById('newsText').value = examples[index];
    hideResults();
    hideError();
}

// Enter key to analyze
document.getElementById('newsText')?.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        analyzeNews();
    }
});

// Check API health on load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('✅ API is healthy');
        }
    } catch (error) {
        console.warn('⚠️ Cannot connect to API. Make sure backend is running.');
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAPIHealth();
    console.log('🚀 Indonesian Fake News Detector initialized');
});
