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
    document.querySelector('.form-container').style.display = 'block';
    document.getElementById('recommendationForm').reset();
    
    // Réinitialiser les valeurs affichées des sliders
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.nextElementSibling.textContent = slider.value;
    });
});
