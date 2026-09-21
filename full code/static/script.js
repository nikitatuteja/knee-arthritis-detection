document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnClear = document.getElementById('btn-clear');
    const btnAnalyze = document.getElementById('btn-analyze');
    
    // States
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');
    
    let currentFile = null;

    // --- File Drag and Drop Handling ---
    
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (e.g., .jpg, .png)');
            return;
        }
        
        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            imagePreviewContainer.classList.remove('hidden');
            resetResults();
        };
        reader.readAsDataURL(file);
    }

    btnClear.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        imagePreviewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        resetResults();
    });

    function resetResults() {
        emptyState.classList.remove('hidden');
        loadingState.classList.add('hidden');
        resultState.classList.add('hidden');
    }

    // --- API Interaction ---
    
    btnAnalyze.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI updates
        emptyState.classList.add('hidden');
        resultState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        btnAnalyze.disabled = true;
        btnAnalyze.textContent = "Analyzing...";

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to analyze image');
            }

            displayResults(data);
        } catch (error) {
            alert(`Analysis Error: ${error.message}`);
            resetResults();
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.textContent = "Run AI Analysis";
            loadingState.classList.add('hidden');
        }
    });

    function displayResults(data) {
        resultState.classList.remove('hidden');
        
        // Set Severity
        const severityValue = document.getElementById('severity-value');
        const severityBadge = document.getElementById('severity-badge');
        
        // Strip numbers if they exist from the dataset labels (e.g. '0Normal')
        const cleanSeverity = data.severity.replace(/^[0-9]+/, '');
        
        severityValue.textContent = cleanSeverity;
        
        // Reset classes
        severityBadge.className = 'severity-badge';
        severityValue.className = '';
        
        // Add dynamic color class
        severityBadge.classList.add(`severity-${cleanSeverity}`);
        severityValue.classList.add(`severity-${cleanSeverity}`);
        
        // Set Confidence
        const confidencePct = Math.round(data.confidence * 100);
        document.getElementById('confidence-value').textContent = `${confidencePct}%`;
        
        // Trigger animation for progress bar
        setTimeout(() => {
            document.getElementById('confidence-fill').style.width = `${confidencePct}%`;
        }, 100);

        // Set Report
        const reportContent = document.getElementById('report-content');
        
        // Basic markdown formatting for the report (newlines to <br>, bullet points)
        let formattedReport = data.report
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>')
            .replace(/\*/g, '•');
            
        reportContent.innerHTML = formattedReport;
    }
});
