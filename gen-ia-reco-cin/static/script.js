// Update slider values
document.querySelectorAll('input[type="range"]').forEach(slider => {
    const valueDisplay = slider.nextElementSibling;
    
    slider.addEventListener('input', (e) => {
        valueDisplay.textContent = e.target.value;
    });
});

// Handle form submission
document.getElementById('recommendationForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form data
    const formData = {
        description: document.getElementById('description').value,
        similar_title: document.getElementById('similar_title').value,
        action_intensity: parseInt(document.getElementById('action').value),
        narrative_complexity: parseInt(document.getElementById('complexity').value),
        darkness: parseInt(document.getElementById('darkness').value),
        realism: parseInt(document.getElementById('realism').value),
        period: document.querySelector('input[name="period"]:checked')?.value || null
    };
    
    // Hide form and show loader
    document.querySelector('.form-container').style.display = 'none';
    document.querySelector('.loader-container').classList.add('active');
    
    try {
        // Display input summary
        displayInputSummary(formData);
        
        // Send request to backend
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error('Error retrieving recommendations');
        }
        
        const data = await response.json();
        
        // Hide loader and display results
        document.querySelector('.loader-container').classList.remove('active');
        displayResults(data.recommendations, data.genai_justification);
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
        document.querySelector('.loader-container').classList.remove('active');
        document.querySelector('.form-container').style.display = 'block';
    }
});

// Display input summary
function displayInputSummary(formData) {
    const summaryContainer = document.getElementById('summaryContent');
    const inputSummary = document.getElementById('inputSummary');
    
    const periodLabels = {
        'present-2020': 'Present - 2020',
        '2020-2015': '2020 - 2015',
        '2015-2010': '2015 - 2010',
        '2010-2000': '2010 - 2000',
        '2000-1980': '2000 - 1980',
        '<1980': 'Before 1980'
    };
    
    const scaleDescriptions = {
        1: '1 - Very little',
        2: '2 - Little',
        3: '3 - Moderate',
        4: '4 - A lot',
        5: '5 - Very much'
    };
    
    summaryContainer.innerHTML = '';
    
    // Description
    if (formData.description) {
        const descItem = document.createElement('div');
        descItem.className = 'summary-item';
        descItem.innerHTML = `<span class="summary-label">📝 Description:</span><span class="summary-value">${formData.description.substring(0, 50)}${formData.description.length > 50 ? '...' : ''}</span>`;
        summaryContainer.appendChild(descItem);
    }
    
    // Similar movie
    if (formData.similar_title) {
        const similarItem = document.createElement('div');
        similarItem.className = 'summary-item';
        similarItem.innerHTML = `<span class="summary-label">🎬 Similar movie:</span><span class="summary-value">${formData.similar_title}</span>`;
        summaryContainer.appendChild(similarItem);
    }
    
    // Action
    const actionItem = document.createElement('div');
    actionItem.className = 'summary-item';
    actionItem.innerHTML = `<span class="summary-label">⚡ Action:</span><span class="summary-value">${scaleDescriptions[formData.action_intensity]}</span>`;
    summaryContainer.appendChild(actionItem);
    
    // Complexity
    const complexityItem = document.createElement('div');
    complexityItem.className = 'summary-item';
    complexityItem.innerHTML = `<span class="summary-label">🧩 Complexity:</span><span class="summary-value">${scaleDescriptions[formData.narrative_complexity]}</span>`;
    summaryContainer.appendChild(complexityItem);
    
    // Darkness
    const darknessItem = document.createElement('div');
    darknessItem.className = 'summary-item';
    darknessItem.innerHTML = `<span class="summary-label">🌑 Darkness:</span><span class="summary-value">${scaleDescriptions[formData.darkness]}</span>`;
    summaryContainer.appendChild(darknessItem);
    
    // Realism
    const realismItem = document.createElement('div');
    realismItem.className = 'summary-item';
    realismItem.innerHTML = `<span class="summary-label">🎭 Realism:</span><span class="summary-value">${scaleDescriptions[formData.realism]}</span>`;
    summaryContainer.appendChild(realismItem);
    
    // Period
    if (formData.period) {
        const periodItem = document.createElement('div');
        periodItem.className = 'summary-item';
        periodItem.innerHTML = `<span class="summary-label">📅 Period:</span><span class="summary-value">${periodLabels[formData.period] || formData.period}</span>`;
        summaryContainer.appendChild(periodItem);
    }
    
    inputSummary.style.display = 'block';
}

// Display results
function displayResults(recommendations, genaiJustification) {
    const resultsContainer = document.querySelector('.results-container');
    const resultsContent = document.getElementById('results-content');
    
    resultsContent.innerHTML = '';
    
    // Display GenAI justification if available
    if (genaiJustification) {
        const justificationBox = document.createElement('div');
        justificationBox.className = 'genai-justification-box';
        justificationBox.innerHTML = `
            <div style="display: flex; align-items: flex-start; gap: 12px;">
                <span style="font-size: 24px;">💡</span>
                <div>
                    <h3 style="margin: 0 0 8px 0; font-size: 18px; color: #6366f1;">Why these recommendations?</h3>
                    <p style="margin: 0; line-height: 1.6; color: #374151; font-size: 14px;">${genaiJustification}</p>
                </div>
            </div>
        `;
        resultsContent.appendChild(justificationBox);
        
        // Add a divider
        const divider = document.createElement('div');
        divider.style.height = '2px';
        divider.style.background = 'linear-gradient(to right, transparent, #e5e7eb, transparent)';
        divider.style.margin = '20px 0';
        resultsContent.appendChild(divider);
    }
    
    recommendations.forEach((movie, index) => {
        const movieCard = document.createElement('div');
        movieCard.className = 'movie-card';
        
        movieCard.innerHTML = `
            <div style="display: flex; align-items: flex-start;">
                <span class="movie-rank">${index + 1}</span>
                <div style="flex: 1;">
                    <h3 class="movie-title">${movie.title} (${movie.year})</h3>
                    <div class="movie-info">
                        <span>📁 ${movie.category}</span>
                        <span>🎭 ${movie.genre}</span>
                        <span class="movie-score">⭐ Score: ${movie.score.toFixed(4)}</span>
                    </div>
                    <p class="movie-description">${movie.description}</p>
                </div>
            </div>
        `;
        
        resultsContent.appendChild(movieCard);
    });
    
    resultsContainer.classList.add('active');
}

// Back button
document.getElementById('backBtn').addEventListener('click', () => {
    document.querySelector('.results-container').classList.remove('active');
    document.getElementById('inputSummary').style.display = 'none';
    document.querySelector('.form-container').style.display = 'block';
    document.getElementById('recommendationForm').reset();
    
    // Reset slider display values
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.nextElementSibling.textContent = slider.value;
    });
});
