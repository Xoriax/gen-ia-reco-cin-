// Mise à jour des valeurs des sliders
document.querySelectorAll('input[type="range"]').forEach(slider => {
    const valueDisplay = slider.nextElementSibling;
    
    slider.addEventListener('input', (e) => {
        valueDisplay.textContent = e.target.value;
    });
});

// Gestion du formulaire
document.getElementById('recommendationForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Récupération des valeurs du formulaire
    const formData = {
        description: document.getElementById('description').value,
        similar_title: document.getElementById('similar_title').value,
        action_intensity: parseInt(document.getElementById('action').value),
        narrative_complexity: parseInt(document.getElementById('complexity').value),
        darkness: parseInt(document.getElementById('darkness').value),
        realism: parseInt(document.getElementById('realism').value),
        period: document.querySelector('input[name="period"]:checked')?.value || null
    };
    
    // Masquer le formulaire et afficher le loader
    document.querySelector('.form-container').style.display = 'none';
    document.querySelector('.loader-container').classList.add('active');
    
    try {
        // Afficher le résumé des inputs
        displayInputSummary(formData);
        
        // Envoi de la requête au backend
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error('Erreur lors de la récupération des recommandations');
        }
        
        const data = await response.json();
        
        // Masquer le loader et afficher les résultats
        document.querySelector('.loader-container').classList.remove('active');
        displayResults(data.recommendations);
        
    } catch (error) {
        console.error('Erreur:', error);
        alert('Une erreur est survenue. Veuillez réessayer.');
        document.querySelector('.loader-container').classList.remove('active');
        document.querySelector('.form-container').style.display = 'block';
    }
});

// Affichage du résumé des inputs
function displayInputSummary(formData) {
    const summaryContainer = document.getElementById('summaryContent');
    const inputSummary = document.getElementById('inputSummary');
    
    const periodLabels = {
        'present-2020': 'Présent - 2020',
        '2020-2015': '2020 - 2015',
        '2015-2010': '2015 - 2010',
        '2010-2000': '2010 - 2000',
        '2000-1980': '2000 - 1980',
        '<1980': '< 1980'
    };
    
    const scaleDescriptions = {
        1: '1 - Très peu',
        2: '2 - Peu',
        3: '3 - Modéré',
        4: '4 - Beaucoup',
        5: '5 - Énormément'
    };
    
    summaryContainer.innerHTML = '';
    
    // Description
    if (formData.description) {
        const descItem = document.createElement('div');
        descItem.className = 'summary-item';
        descItem.innerHTML = `<span class="summary-label">📝 Description:</span><span class="summary-value">${formData.description.substring(0, 50)}${formData.description.length > 50 ? '...' : ''}</span>`;
        summaryContainer.appendChild(descItem);
    }
    
    // Film similaire
    if (formData.similar_title) {
        const similarItem = document.createElement('div');
        similarItem.className = 'summary-item';
        similarItem.innerHTML = `<span class="summary-label">🎬 Film similaire:</span><span class="summary-value">${formData.similar_title}</span>`;
        summaryContainer.appendChild(similarItem);
    }
    
    // Action
    const actionItem = document.createElement('div');
    actionItem.className = 'summary-item';
    actionItem.innerHTML = `<span class="summary-label">⚡ Action:</span><span class="summary-value">${scaleDescriptions[formData.action_intensity]}</span>`;
    summaryContainer.appendChild(actionItem);
    
    // Complexité
    const complexityItem = document.createElement('div');
    complexityItem.className = 'summary-item';
    complexityItem.innerHTML = `<span class="summary-label">🧩 Complexité:</span><span class="summary-value">${scaleDescriptions[formData.narrative_complexity]}</span>`;
    summaryContainer.appendChild(complexityItem);
    
    // Noirceur
    const darknessItem = document.createElement('div');
    darknessItem.className = 'summary-item';
    darknessItem.innerHTML = `<span class="summary-label">🌑 Noirceur:</span><span class="summary-value">${scaleDescriptions[formData.darkness]}</span>`;
    summaryContainer.appendChild(darknessItem);
    
    // Réalisme
    const realismItem = document.createElement('div');
    realismItem.className = 'summary-item';
    realismItem.innerHTML = `<span class="summary-label">🎭 Réalisme:</span><span class="summary-value">${scaleDescriptions[formData.realism]}</span>`;
    summaryContainer.appendChild(realismItem);
    
    // Période
    if (formData.period) {
        const periodItem = document.createElement('div');
        periodItem.className = 'summary-item';
        periodItem.innerHTML = `<span class="summary-label">📅 Période:</span><span class="summary-value">${periodLabels[formData.period] || formData.period}</span>`;
        summaryContainer.appendChild(periodItem);
    }
    
    inputSummary.style.display = 'block';
}

// Affichage des résultats
function displayResults(recommendations) {
    const resultsContainer = document.querySelector('.results-container');
    const resultsContent = document.getElementById('results-content');
    
    resultsContent.innerHTML = '';
    
    recommendations.forEach((movie, index) => {
        const movieCard = document.createElement('div');
        movieCard.className = 'movie-card';
        
        movieCard.innerHTML = `
            <div style="display: flex; align-items: flex-start;">
                <span class="movie-rank">${index + 1}</span>
                <div style="flex: 1;">
                    <h3 class="movie-title">${movie.titre} (${movie.année})</h3>
                    <div class="movie-info">
                        <span>📁 ${movie.catégorie}</span>
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

// Bouton retour
document.getElementById('backBtn').addEventListener('click', () => {
    document.querySelector('.results-container').classList.remove('active');
    document.getElementById('inputSummary').style.display = 'none';
    document.querySelector('.form-container').style.display = 'block';
    document.getElementById('recommendationForm').reset();
    
    // Réinitialiser les valeurs affichées des sliders
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.nextElementSibling.textContent = slider.value;
    });
});
