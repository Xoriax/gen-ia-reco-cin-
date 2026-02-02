// Conversation Flow Logic
class ConversationalQuiz {
    constructor() {
        this.currentStep = 0;
        this.answers = {
            description: '',
            similar_title: '',
            action_intensity: 3,
            narrative_complexity: 3,
            darkness: 3,
            realism: 3,
            period: ''
        };

        this.steps = [
            {
                id: 'description',
                question: '📝 What are you in the mood for?',
                subtext: 'Describe the type of movie or show you want to watch',
                type: 'textarea',
                placeholder: 'E.g., An intense drama with suspense and crime...',
                required: true
            },
            {
                id: 'similar_title',
                question: '🎬 Any similar movies you like? (Optional)',
                subtext: 'Give us a title of a movie you enjoy',
                type: 'text',
                placeholder: 'E.g., The Shawshank Redemption, Inception...',
                required: false
            },
            {
                id: 'action_intensity',
                question: '⚡ How much action do you want?',
                subtext: 'From calm and contemplative to explosive action',
                type: 'slider',
                min: 1,
                max: 5,
                labels: ['Very Calm', 'Pure Action'],
                valueLabels: ['', 'Very Calm', 'Calm', 'Balanced', 'Action-packed', 'Explosive']
            },
            {
                id: 'narrative_complexity',
                question: '🧩 How complex should the plot be?',
                subtext: 'From simple and straightforward to mind-bending',
                type: 'slider',
                min: 1,
                max: 5,
                labels: ['Simple Plot', 'Mind-bending'],
                valueLabels: ['', 'Simple', 'Simple', 'Moderate', 'Complex', 'Mind-bending']
            },
            {
                id: 'darkness',
                question: '🌑 How dark or violent?',
                subtext: 'From family-friendly to dark and gritty',
                type: 'slider',
                min: 1,
                max: 5,
                labels: ['Family-friendly', 'Dark/Violent'],
                valueLabels: ['', 'Family-friendly', 'Light', 'Balanced', 'Dark', 'Very Dark']
            },
            {
                id: 'realism',
                question: '🌍 Realistic or fantastical?',
                subtext: 'From grounded documentaries to surreal fantasy',
                type: 'slider',
                min: 1,
                max: 5,
                labels: ['Documentary', 'Fantasy'],
                valueLabels: ['', 'Grounded', 'Realistic', 'Balanced', 'Fantastical', 'Surreal']
            },
            {
                id: 'period',
                question: '📅 Which time period do you prefer?',
                subtext: 'Choose a release period or select all',
                type: 'radio',
                options: [
                    { value: 'present-2020', label: '🆕 Present - 2020' },
                    { value: '2020-2015', label: '📺 2020 - 2015' },
                    { value: '2015-2010', label: '🎬 2015 - 2010' },
                    { value: '2010-2000', label: '🎞️ 2010 - 2000' },
                    { value: '2000-1980', label: '📽️ 2000 - 1980' },
                    { value: '<1980', label: '🎭 Before 1980' },
                    { value: '', label: '⏰ Any period' }
                ]
            }
        ];

        this.init();
    }

    init() {
        this.renderCurrentStep();
        this.attachEventListeners();
    }

    renderCurrentStep() {
        const step = this.steps[this.currentStep];
        const questionText = document.getElementById('questionText');
        const questionSubtext = document.getElementById('questionSubtext');
        const inputArea = document.getElementById('inputArea');
        const chatHistory = document.getElementById('chatHistory');
        const nextBtn = document.getElementById('nextBtn');
        const backBtn = document.getElementById('backBtn');

        // Update progress
        const progress = Math.round(((this.currentStep + 1) / this.steps.length) * 100);
        document.querySelector('.conversation-container').style.backgroundImage = 
            `linear-gradient(to right, rgba(102, 126, 234, 0.1) 0%, rgba(102, 126, 234, 0.1) ${progress}%, transparent ${progress}%, transparent 100%)`;

        // Update question text
        questionText.textContent = step.question;
        questionSubtext.textContent = step.subtext;

        // Clear previous input
        inputArea.innerHTML = '';

        // Render input based on type
        if (step.type === 'textarea') {
            const textarea = document.createElement('textarea');
            textarea.id = `input_${step.id}`;
            textarea.placeholder = step.placeholder;
            textarea.value = this.answers[step.id];
            inputArea.appendChild(textarea);
        } else if (step.type === 'text') {
            const input = document.createElement('input');
            input.type = 'text';
            input.id = `input_${step.id}`;
            input.placeholder = step.placeholder;
            input.value = this.answers[step.id];
            inputArea.appendChild(input);
        } else if (step.type === 'slider') {
            const sliderGroup = document.createElement('div');
            sliderGroup.className = 'slider-group';

            const label = document.createElement('div');
            label.className = 'slider-label';
            label.innerHTML = `<span></span><span class="slider-value">${step.valueLabels[this.answers[step.id]]}</span>`;
            sliderGroup.appendChild(label);

            const slider = document.createElement('input');
            slider.type = 'range';
            slider.id = `input_${step.id}`;
            slider.className = 'slider-range';
            slider.min = step.min;
            slider.max = step.max;
            slider.value = this.answers[step.id];
            slider.addEventListener('input', (e) => {
                label.querySelector('.slider-value').textContent = step.valueLabels[e.target.value];
            });
            sliderGroup.appendChild(slider);

            const sliderLabels = document.createElement('div');
            sliderLabels.className = 'slider-labels';
            sliderLabels.innerHTML = `<span>${step.labels[0]}</span><span>${step.labels[1]}</span>`;
            sliderGroup.appendChild(sliderLabels);

            inputArea.appendChild(sliderGroup);
        } else if (step.type === 'radio') {
            const optionsGroup = document.createElement('div');
            optionsGroup.className = 'options-group';

            step.options.forEach(option => {
                const optionItem = document.createElement('div');
                optionItem.className = 'option-item';

                const radio = document.createElement('input');
                radio.type = 'radio';
                radio.name = `input_${step.id}`;
                radio.id = `input_${step.id}_${option.value}`;
                radio.value = option.value;
                radio.checked = this.answers[step.id] === option.value;

                const label = document.createElement('label');
                label.htmlFor = `input_${step.id}_${option.value}`;
                label.textContent = option.label;

                optionItem.appendChild(radio);
                optionItem.appendChild(label);
                optionsGroup.appendChild(optionItem);
            });

            inputArea.appendChild(optionsGroup);
        }

        // Update button states
        backBtn.style.display = this.currentStep > 0 ? 'block' : 'none';
        nextBtn.textContent = this.currentStep === this.steps.length - 1 ? '🎬 Get Recommendations →' : 'Continue →';
    }

    attachEventListeners() {
        document.getElementById('nextBtn').addEventListener('click', () => this.nextStep());
        document.getElementById('backBtn').addEventListener('click', () => this.prevStep());
        document.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) this.nextStep();
        });
    }

    nextStep() {
        const step = this.steps[this.currentStep];
        const input = document.getElementById(`input_${step.id}`);

        // Validate
        if (step.required && !input.value.trim()) {
            alert('Please fill in this field');
            return;
        }

        // Get answer based on type
        if (step.type === 'radio') {
            const checked = document.querySelector(`input[name="input_${step.id}"]:checked`);
            this.answers[step.id] = checked ? checked.value : '';
        } else if (step.type === 'slider') {
            this.answers[step.id] = parseInt(input.value);
        } else {
            this.answers[step.id] = input.value.trim();
        }

        // Add to chat history
        this.addToHistory(step);

        // Move to next step or show results
        if (this.currentStep < this.steps.length - 1) {
            this.currentStep++;
            this.renderCurrentStep();
        } else {
            this.showResults();
        }
    }

    prevStep() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.renderCurrentStep();
        }
    }

    addToHistory(step) {
        const chatHistory = document.getElementById('chatHistory');

        // Bot question
        const botMsg = document.createElement('div');
        botMsg.className = 'chat-message bot';
        botMsg.innerHTML = `<div class="message-label">Quiz</div><div class="message-bubble">${step.question}</div>`;
        chatHistory.appendChild(botMsg);

        // User answer
        const userMsg = document.createElement('div');
        userMsg.className = 'chat-message user';
        const answer = this.formatAnswer(step);
        userMsg.innerHTML = `<div class="message-label">You</div><div class="message-bubble">${answer}</div>`;
        chatHistory.appendChild(userMsg);

        // Scroll to bottom
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    formatAnswer(step) {
        const answer = this.answers[step.id];
        if (!answer) return '(skipped)';

        if (step.type === 'slider') {
            return `${step.valueLabels[answer]} (${answer}/5)`;
        } else if (step.type === 'radio') {
            const option = step.options.find(o => o.value === answer);
            return option ? option.label : answer;
        } else {
            return answer.substring(0, 50) + (answer.length > 50 ? '...' : '');
        }
    }

    async showResults() {
        document.getElementById('currentQuestion').style.display = 'none';
        document.querySelector('.button-group').style.display = 'none';

        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'block';

        // Show Q&A Summary
        this.displayQASummary();

        // Fetch recommendations
        await this.fetchRecommendations();
    }

    displayQASummary() {
        const qaSummary = document.getElementById('qaSummary');
        qaSummary.innerHTML = '<h3>📋 Your Preferences</h3>';

        const questionLabels = {
            description: 'What you\'re looking for',
            similar_title: 'Similar movie',
            action_intensity: 'Action level',
            narrative_complexity: 'Plot complexity',
            darkness: 'Darkness/Violence',
            realism: 'Realism vs Fantasy',
            period: 'Time period'
        };

        Object.keys(this.answers).forEach(key => {
            const step = this.steps.find(s => s.id === key);
            const answer = this.answers[key];

            if (answer) {
                const qaItem = document.createElement('div');
                qaItem.className = 'qa-item';

                const formattedAnswer = this.formatAnswer(step);
                qaItem.innerHTML = `
                    <div class="qa-question">${questionLabels[key]}</div>
                    <div class="qa-answer">${formattedAnswer}</div>
                `;

                qaSummary.appendChild(qaItem);
            }
        });
    }

    async fetchRecommendations() {
        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    description: this.answers.description,
                    similar_title: this.answers.similar_title,
                    action_intensity: this.answers.action_intensity,
                    narrative_complexity: this.answers.narrative_complexity,
                    darkness: this.answers.darkness,
                    realism: this.answers.realism,
                    period: this.answers.period
                })
            });

            const data = await response.json();

            if (!data.success && data.error) {
                alert('Error: ' + data.error);
                return;
            }

            this.displayRecommendations(data.recommendations, data.genai_justification);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        }
    }

    displayRecommendations(recommendations, justification) {
        // Justification
        if (justification) {
            const justBox = document.getElementById('justificationBox');
            justBox.innerHTML = `
                <h3>💡 Why These Recommendations?</h3>
                <p>${justification}</p>
            `;
        }

        // Movies Grid
        const grid = document.getElementById('recommendationsGrid');
        grid.innerHTML = '';

        recommendations.forEach((movie, index) => {
            const card = document.createElement('div');
            card.className = 'movie-card';
            card.innerHTML = `
                <div class="movie-rank">${index + 1}</div>
                <h3 class="movie-title">${movie.title}</h3>
                <div class="movie-year">📅 ${movie.year}</div>
                <div class="movie-info">
                    <span>📁 ${movie.category}</span>
                    <span>🎭 ${movie.genre}</span>
                    <span class="movie-score">⭐ ${movie.score.toFixed(2)}</span>
                </div>
                <p class="movie-description">${movie.description}</p>
            `;
            grid.appendChild(card);
        });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ConversationalQuiz();
});
