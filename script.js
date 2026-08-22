
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const uploadContainer = document.getElementById('upload-container');
const previewContainer = document.getElementById('preview-container');
const loadingContainer = document.getElementById('loading-container');
const resultContainer = document.getElementById('result-container');
const imagePreview = document.getElementById('image-preview');
const imageAnalyzing = document.getElementById('image-analyzing');
const analyzeBtn = document.getElementById('analyze-btn');
const changeImageBtn = document.getElementById('change-image-btn');
const resetBtn = document.getElementById('reset-btn');
const probBars = document.getElementById('probability-bars');
const historySection = document.getElementById('history-section');
const historyList = document.getElementById('history-list');

// Custom Cursor Logic
const cursorDot = document.getElementById('cursor-dot');
const cursorOutline = document.getElementById('cursor-outline');

window.addEventListener('mousemove', (e) => {
    const posX = e.clientX;
    const posY = e.clientY;

    cursorDot.style.left = `${posX}px`;
    cursorDot.style.top = `${posY}px`;

    // Outline follows with a slight delay
    cursorOutline.animate({
        left: `${posX}px`,
        top: `${posY}px`
    }, { duration: 500, fill: 'forwards' });
});

// Cursor Hover Effects
const addCursorHover = () => {
    document.querySelectorAll('a, button, .drop-zone, .glass-card').forEach(el => {
        el.addEventListener('mouseenter', () => document.body.classList.add('cursor-hover'));
        el.addEventListener('mouseleave', () => document.body.classList.remove('cursor-hover'));
    });
};

// 3D Tilt Effect
const addTiltEffect = () => {
    document.querySelectorAll('.glass-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;
            
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg)`;
        });
    });
};

// Handle Splash Screen
window.addEventListener('load', () => {
    const splash = document.getElementById('splash-screen');
    setTimeout(() => {
        splash.classList.add('fade-out');
        addCursorHover();
        addTiltEffect();
    }, 1000); // Show splash for at least 1s
});

// Reveal Animations on Scroll
const revealElements = () => {
    const reveals = document.querySelectorAll('.reveal');
    reveals.forEach(element => {
        const windowHeight = window.innerHeight;
        const elementTop = element.getBoundingClientRect().top;
        const elementVisible = 150;
        if (elementTop < windowHeight - elementVisible) {
            element.classList.add('active');
        }
    });
};
window.addEventListener('scroll', revealElements);
// Initial check
document.addEventListener('DOMContentLoaded', () => {
    revealElements();
    updateHistoryUI();
});

// Clear history on refresh/reload as requested
let history = [];

function updateHistoryUI() {
    if (history.length === 0) {
        historySection.classList.add('hidden');
        return;
    }
    historySection.classList.remove('hidden');
    historyList.innerHTML = '';
    history.forEach(item => {
        const card = `
            <div class="glass-card p-6 rounded-3xl border border-slate-100/10 shadow-sm hover:shadow-xl transition-all duration-300 reveal active">
                <div class="flex justify-between items-start mb-4">
                    <div class="text-[10px] font-bold text-slate-400 uppercase tracking-widest">${item.date}</div>
                    <div class="px-3 py-1 rounded-full text-[10px] font-bold text-white uppercase tracking-widest ${severityColors[item.class]}">${item.class}</div>
                </div>
                <h4 class="font-bold text-white mb-2 truncate">${item.fileName}</h4>
                <div class="flex items-center justify-between text-xs">
                    <span class="text-slate-400">Confidence</span>
                    <span class="font-bold text-white">${(item.confidence * 100).toFixed(1)}%</span>
                </div>
            </div>
        `;
        historyList.insertAdjacentHTML('beforeend', card);
    });
}
let selectedFile = null;

// Colors for severity levels
const severityColors = {
    'Normal': 'bg-green-500',
    'Doubtful': 'bg-yellow-400',
    'Mild': 'bg-orange-400',
    'Moderate': 'bg-red-500',
    'Severe': 'bg-slate-900'
};

const severityReasoning = {
    'Normal': 'The AI detected healthy joint spaces and smooth bone surfaces with no evidence of osteophytes or narrowing.',
    'Doubtful': 'Possible subtle joint space narrowing and minute osteophyte formation were identified in the early analysis pass.',
    'Mild': 'Definite osteophytes and possible narrowing of joint space were detected, indicating early-stage arthritic changes.',
    'Moderate': 'Analysis shows multiple moderate osteophytes, definite joint space narrowing, and some subchondral sclerosis.',
    'Severe': 'Large osteophytes, marked narrowing of the joint space, and significant subchondral sclerosis were identified, indicating advanced degeneration.'
};

// Handle Splash Screendrop zone click
dropZone.addEventListener('click', () => fileInput.click());

// Handle file selection
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

// Drag and drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('active');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('active');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file (PNG/JPG).');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imageAnalyzing.src = e.target.result;
        uploadContainer.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        resultContainer.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

changeImageBtn.addEventListener('click', () => {
    fileInput.click();
});

// API Configuration
// When deploying, change this to your production backend URL (e.g., Render or Railway)
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? '' 
    : 'https://kneearthritisdetection.onrender.com';

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    previewContainer.classList.add('hidden');
    loadingContainer.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Analysis failed');

        const data = await response.json();
        
        // Update history array for current session
        const historyItem = {
            id: Date.now(),
            date: new Date().toLocaleDateString(),
            class: data.class,
            confidence: data.confidence,
            fileName: selectedFile.name
        };
        history.unshift(historyItem);
        updateHistoryUI();
        
        setTimeout(() => displayResult(data), 1500); // Small delay for "analysis" effect
    } catch (error) {
        console.error('Error:', error);
        alert('Error connecting to the analysis engine. Is the backend running?');
        loadingContainer.classList.add('hidden');
        previewContainer.classList.remove('hidden');
    }
});

function displayResult(data) {
    loadingContainer.classList.add('hidden');
    resultContainer.classList.remove('hidden');

    const badge = document.getElementById('result-severity-badge');
    const title = document.getElementById('result-title');
    const confidence = document.getElementById('result-confidence');
    const reasoning = document.getElementById('result-reasoning');

    badge.className = `inline-block px-6 py-2 rounded-full text-white font-bold text-sm uppercase tracking-widest shadow-lg ${severityColors[data.class]}`;
    badge.innerText = data.class;
    
    title.innerText = `${data.class} Arthritis Detected`;
    confidence.innerText = `AI Confidence Level: ${(data.confidence * 100).toFixed(1)}%`;
    reasoning.innerText = severityReasoning[data.class] || 'Detailed analysis complete.';

    // Probability bars
    probBars.innerHTML = '';
    Object.entries(data.all_predictions).forEach(([label, prob]) => {
        const percentage = (prob * 100).toFixed(1);
        const barHtml = `
            <div class="space-y-1">
                <div class="flex justify-between text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                    <span>${label}</span>
                    <span class="text-slate-300">${percentage}%</span>
                </div>
                <div class="w-full bg-white/5 rounded-full h-1.5 overflow-hidden border border-white/5">
                    <div class="h-full ${severityColors[label]} transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(99,102,241,0.3)]" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
        probBars.insertAdjacentHTML('beforeend', barHtml);
    });
}

resetBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    resultContainer.classList.add('hidden');
    uploadContainer.classList.remove('hidden');
});
