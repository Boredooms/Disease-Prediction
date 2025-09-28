// ===== PARTICLE SYSTEM =====
function createParticles() {
    const container = document.getElementById('particles');
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 20 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
        container.appendChild(particle);
    }
}

// ===== FILE HANDLING =====
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');

if (uploadBtn) {
    uploadBtn.addEventListener('click', () => fileInput.click());
}

if (uploadArea) {
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
}

if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

// ===== FILE UPLOAD HANDLER =====
async function handleFileUpload(file) {
    if (!file) return;

    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG, PNG) or PDF.');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB.');
        return;
    }

    showLoading();

    const formData = new FormData();
    formData.append('prescription', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (response.ok) {
            displayResults(result);
        } else {
            throw new Error(result.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing file: ' + error.message);
    } finally {
        hideLoading();
    }
}

// ===== DEMO FUNCTIONALITY =====
const demoInput = document.getElementById('demoInput');
const demoBtn = document.getElementById('demoBtn');

if (demoBtn) {
    demoBtn.addEventListener('click', handleDemo);
}

if (demoInput) {
    demoInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleDemo();
        }
    });
}

async function handleDemo() {
    const medicine = demoInput.value.trim();
    if (!medicine) {
        alert('Please enter a medicine name.');
        return;
    }

    showLoading();

    try {
        const response = await fetch('/demo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ medicine: medicine })
        });

        const result = await response.json();
        
        if (response.ok) {
            displayDemoResults(result);
        } else {
            throw new Error(result.error || 'Demo failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error in demo: ' + error.message);
    } finally {
        hideLoading();
    }
}

// ===== LOADING MANAGEMENT =====
function showLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'flex';
        
        // Simulate stage progression
        const stages = ['stage1', 'stage2', 'stage3', 'stage4'];
        stages.forEach((stageId, index) => {
            setTimeout(() => {
                const stage = document.getElementById(stageId);
                if (stage) {
                    stage.classList.add('active');
                    if (index > 0) {
                        const prevStage = document.getElementById(stages[index - 1]);
                        if (prevStage) {
                            prevStage.classList.remove('active');
                            prevStage.classList.add('completed');
                        }
                    }
                }
            }, index * 1500);
        });
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
        
        // Reset stages
        const stages = ['stage1', 'stage2', 'stage3', 'stage4'];
        stages.forEach(stageId => {
            const stage = document.getElementById(stageId);
            if (stage) {
                stage.classList.remove('active', 'completed');
            }
        });
    }
}

// ===== RESULTS DISPLAY =====
function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const pipelineStages = document.getElementById('pipelineStages');
    const predictionsContainer = document.getElementById('predictionsContainer');

    if (!resultsSection || !pipelineStages || !predictionsContainer) return;

    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // Display pipeline stages
    pipelineStages.innerHTML = `
        <div class="stage-card">
            <div class="stage-header">
                <div class="stage-number">1</div>
                <h4>OCR Processing</h4>
            </div>
            <p><strong>Detected Text:</strong> ${data.stages.ocr.detected_text || 'No text detected'}</p>
            <p><strong>Confidence:</strong> ${(data.stages.ocr.confidence * 100).toFixed(1)}%</p>
            <p><strong>Processing Time:</strong> ${data.stages.ocr.processing_time.toFixed(2)}s</p>
        </div>
        
        <div class="stage-card">
            <div class="stage-header">
                <div class="stage-number">2</div>
                <h4>Medicine Enhancement</h4>
            </div>
            <p><strong>Enhanced Medicines:</strong> ${data.stages.medicine_enhancement.count}</p>
            ${data.stages.medicine_enhancement.enhanced_medicines.map(med => 
                `<p>• ${med.medicine} (${(med.confidence * 100).toFixed(1)}%)</p>`
            ).join('')}
        </div>
        
        <div class="stage-card">
            <div class="stage-header">
                <div class="stage-number">3</div>
                <h4>Clinical BERT</h4>
            </div>
            <p><strong>Medicines Analyzed:</strong> ${data.stages.clinical_bert.medications_count || 0}</p>
            <p><strong>Total Predictions:</strong> ${data.stages.disease_prediction.total_predictions}</p>
            <p><strong>Unique Medicines:</strong> ${Array.isArray(data.stages.clinical_bert.medications) && data.stages.clinical_bert.medications.length > 0 ? data.stages.clinical_bert.medications.join(', ') : 'None'}</p>
            <p><strong>Medical Conditions Found:</strong> ${data.stages.clinical_bert.conditions_count || 0}</p>
            <p><strong>Conditions:</strong> ${Array.isArray(data.stages.clinical_bert.medical_conditions) && data.stages.clinical_bert.medical_conditions.length > 0 ? data.stages.clinical_bert.medical_conditions.join(', ') : 'None'}</p>
        </div>
        

    `;

    // Display predictions
    if (data.summary.top_predictions && data.summary.top_predictions.length > 0) {
        predictionsContainer.innerHTML = `
            <h3 class="text-gradient">
                <i class="fas fa-heartbeat"></i> Disease Predictions
            </h3>
            <div class="predictions-grid">
                ${data.summary.top_predictions.map(pred => `
                    <div class="prediction-card">
                        <h4>${pred.disease}</h4>
                        <p>Model: ${pred.model_name}</p>
                        <p>Medicine: ${pred.medicine}</p>
                        <p><strong>Confidence: ${(pred.confidence * 100).toFixed(1)}%</strong></p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${pred.confidence * 100}%"></div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    } else {
        predictionsContainer.innerHTML = `
            <div class="no-predictions">
                <i class="fas fa-info-circle"></i>
                <h3>No Disease Predictions Found</h3>
                <p>No matching medicines were found in our disease prediction models.</p>
            </div>
        `;
    }
}

function displayDemoResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const pipelineStages = document.getElementById('pipelineStages');
    const predictionsContainer = document.getElementById('predictionsContainer');

    if (!resultsSection || !pipelineStages || !predictionsContainer) return;

    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // Display demo info
    pipelineStages.innerHTML = `
        <div class="stage-card full-width">
            <div class="stage-header">
                <div class="stage-number">
                    <i class="fas fa-flask"></i>
                </div>
                <h4>Demo Prediction for: ${data.medicine}</h4>
            </div>
            <p><strong>Total Predictions:</strong> ${data.total_predictions}</p>
            <p>This is a demo showing disease predictions for the entered medicine.</p>
        </div>
    `;

    // Display predictions
    if (data.predictions && data.predictions.length > 0) {
        predictionsContainer.innerHTML = `
            <h3 class="text-gradient">
                <i class="fas fa-heartbeat"></i> Disease Predictions
            </h3>
            <div class="predictions-grid">
                ${data.predictions.map(pred => `
                    <div class="prediction-card">
                        <h4>${pred.disease}</h4>
                        <p>Model: ${pred.model_name}</p>
                        <p>Medicine: ${pred.medicine}</p>
                        <p><strong>Confidence: ${(pred.confidence * 100).toFixed(1)}%</strong></p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${pred.confidence * 100}%"></div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    } else {
        predictionsContainer.innerHTML = `
            <div class="no-predictions">
                <i class="fas fa-info-circle"></i>
                <h3>No Predictions Found</h3>
                <p>No disease predictions available for "${data.medicine}".</p>
                <p>Try medicines like: Amlodipine, Metformin, Aspirin, Lisinopril</p>
            </div>
        `;
    }
}

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    createParticles();
    
    // Add smooth scroll behavior
    document.documentElement.style.scrollBehavior = 'smooth';
});