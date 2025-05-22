// Component generation functions

function createBentoCard(title, description, linkId, iconName = 'article') {
    const card = document.createElement('a'); 
    card.href = `#${linkId}`;
    card.className = 'bento-card group block transform transition-all duration-300 ease-out hover:shadow-bento-hover hover:-translate-y-1';
    card.setAttribute('data-target-id', linkId); 

    card.innerHTML = `
        <div class="bento-card-accent"></div>
        <div class="bento-card-content p-5 md:p-6">
            <div class="flex items-start gap-3 mb-3">
                <span class="material-icons text-primary text-2xl mt-1">${iconName}</span>
                <div>
                    <h3 class="bento-card-title font-chinese">${title}</h3>
                    <p class="bento-card-description font-sans">${description}</p>
                </div>
            </div>
            <div class="mt-auto pt-3 text-right">
                <span class="text-primary font-medium text-sm group-hover:underline flex items-center justify-end">
                    前往学习 <span class="material-icons text-base ml-1">arrow_forward</span>
                </span>
            </div>
        </div>
    `;
    card.addEventListener('click', (e) => {
        e.preventDefault();
        scrollToElement(linkId); 
    });
    return card;
}

function createKeyConceptCard(title, content, iconName = 'lightbulb_outline') {
    const card = document.createElement('div');
    // Removed 'fade-in', animation will be handled by 'anim-on-scroll' added in processKeyConceptCards
    card.className = 'key-concept-card my-6'; 

    card.innerHTML = `
        <h4 class="key-concept-card-title">
            <span class="material-icons">${iconName}</span>
            <span class="font-chinese">${title}</span>
        </h4>
        <div class="text-sm leading-relaxed">${content}</div>
    `;
    return card;
}


function createStatsCounter(stats) { 
    const statsContainer = document.createElement('div');
    // Removed 'fade-in', animation handled by 'anim-on-scroll' in processStatsBlocks
    statsContainer.className = 'stats-container'; 

    stats.forEach((stat, index) => {
        const statItem = document.createElement('div');
        statItem.className = 'stats-item anim-child'; // Add anim-child for potential stagger from parent
        // statItem.style.setProperty('--fade-in-delay', index); // Delay handled by Framer Motion stagger if parent has it

        statItem.innerHTML = `
            <div class="stats-value">
                <span class="count-up ultra-large-number" data-value="${stat.value}">0</span>
                ${stat.suffix ? `<span class="ultra-large-number-suffix">${stat.suffix}</span>` : ''}
            </div>
            <div class="stats-label font-chinese">${stat.label}</div>
        `;
        statsContainer.appendChild(statItem);
    });
    return statsContainer; 
}

function createFeatureList(title, features) { 
    // The root element will get anim-on-scroll attributes from processFeatureBlocks in utils.js
    let featureHTML = `
        <div class="feature-list-container my-8"> 
            <h4 class="feature-list-title font-chinese">${title}</h4>
            <div class="feature-list-grid">
    `;

    features.forEach((feature, index) => {
        // Add anim-child for potential stagger and individual animation
        featureHTML += `
            <div class="feature-item anim-child">
                <span class="material-icons text-primary mr-4 mt-1">${feature.icon || 'check_circle_outline'}</span>
                <div>
                    <h5 class="font-semibold font-chinese">${feature.title}</h5>
                    <p class="text-gray-600 text-sm font-sans">${feature.description}</p>
                </div>
            </div>
        `;
    });

    featureHTML += `
            </div>
        </div>
    `;
    return featureHTML; 
}

function createQuoteCard(quoteText, author, iconName = 'format_quote') {
    const card = document.createElement('div');
    // Class already includes anim-on-scroll from where it's called or defined in HTML
    // It inherits bento-visual-card styling, which is good.
    // Spans are set in index.html where it's placed.
    card.className = 'quote-card flex flex-col items-center justify-center p-6 text-center'; 

    card.innerHTML = `
        <span class="material-icons text-primary text-4xl mb-4">${iconName}</span>
        <p class="text-lg italic text-gray-700 mb-3 leading-relaxed font-serif">"${quoteText}"</p>
        ${author ? `<p class="text-sm font-semibold text-darkText font-chinese">- ${author}</p>` : ''}
    `;
    return card;
}

