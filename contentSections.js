// Handles loading content into sections and post-processing DOM elements

async function loadAndProcessAllContent() {
    const sectionsConfig = [
        { id: 'introduction', file: 'content/introduction.md' },
        { id: 'algorithms', file: 'content/algorithms.md' },
        { id: 'technologies', file: 'content/technologies.md' },
        { id: 'projects', file: 'content/projects.md' },
        { id: 'advanced', file: 'content/advanced.md' },
        { id: 'conclusion', file: 'content/conclusion.md' }
    ];

    let sectionFadeInDelayCounter = 0;

    for (const section of sectionsConfig) {
        const contentContainer = document.getElementById(`${section.id}-content`);
        const sectionElement = document.getElementById(`${section.id}-section`); 

        if (!contentContainer) {
            console.warn(`Content container for section '${section.id}-content' not found.`);
            continue;
        }

        if (sectionElement) { 
            sectionElement.dataset.animationDelay = (sectionFadeInDelayCounter * 0.05).toString();
            sectionFadeInDelayCounter++;
        }

        contentContainer.innerHTML = '<p class="text-center py-10 text-gray-500">Loading content...</p>';

        try {
            console.log(`Loading content for section: ${section.id}`);
            const rawText = await fetchTextContent(section.file);
            
            if (!rawText || typeof rawText !== 'string') {
                throw new Error(`Invalid content format for ${section.file}`);
            }

            console.log(`Parsing markdown for section: ${section.id}`);
            const { metadata, content: htmlContent } = parseMarkdown(rawText);
            
            if (!htmlContent || typeof htmlContent !== 'string') {
                throw new Error(`Failed to parse markdown for ${section.file}`);
            }

            console.log(`Setting content for section: ${section.id}`);
            contentContainer.innerHTML = htmlContent;

            // 确保内容加载后再处理特殊元素
            if (contentContainer.innerHTML.trim()) {
                console.log(`Processing special elements for section: ${section.id}`);
                processKeyConceptCards(contentContainer);
                processSVGDiagrams(contentContainer);
            } else {
                throw new Error(`Empty content after processing ${section.file}`);
            }

        } catch (error) {
            console.error(`Error loading or processing content for section ${section.id}:`, error);
            contentContainer.innerHTML = `<div class="p-6 bg-red-100 text-red-700 rounded-lg border border-red-300">
                <h3 class="font-bold text-lg mb-2">Error Loading Section Content</h3>
                <p>Could not load content for: ${section.file}</p>
                <p class="text-sm mt-1">Details: ${error.message}</p>
            </div>`;
        }
    }
}


function processKeyConceptCards(container) {
    container.querySelectorAll('blockquote').forEach(blockquote => {
        if (blockquote.closest('.key-concept-card')) return;

        let title = 'Key Concept'; 
        let icon = 'lightbulb_outline'; 
        let blockquoteContent = blockquote.innerHTML;

        const firstP = blockquote.querySelector('p:first-child');
        if (firstP) {
            const firstPText = firstP.textContent.trim();
            const titleMatch = firstPText.match(/^\*\*(.*?)(?:\s*\(([^)]+)\))?:\*\*(.*)/i);

            if (titleMatch) {
                title = titleMatch[1].trim();
                if (titleMatch[2]) icon = titleMatch[2].trim().toLowerCase().replace(/\s+/g, '_');
                
                if (titleMatch[3] && titleMatch[3].trim() !== "") {
                     firstP.innerHTML = titleMatch[3].trim(); 
                } else {
                    firstP.remove(); 
                    blockquoteContent = blockquote.innerHTML; 
                }
            }
        }
        
        const card = createKeyConceptCard(title, blockquoteContent, icon); 
        // Add animation attributes to the key concept card itself
        card.classList.add('anim-on-scroll');
        card.dataset.animationType = 'fadeInUp'; 
        // card.dataset.animationDelay = '0.1'; // Or a staggered delay if needed

        blockquote.parentNode.replaceChild(card, blockquote);
    });
}


function processSVGDiagrams(container) {
    // Process text markers for specific, pre-defined SVG illustrations
    const placeholderParagraphs = Array.from(container.querySelectorAll('p'));
    placeholderParagraphs.forEach(p => {
        const text = p.textContent.trim();
        let svgString = null;
        let caption = text; 

        if (text.includes("Agent 的感知-决策-行动循环")) {
            svgString = createAgentCycleIllustration(); // from illustrations.js
            caption = "图：Agent 的感知-决策-行动循环";
        } else if (text.includes("Transformer 架构的简化示意图") || text.includes("Transformer 架构")) {
            svgString = createTransformerIllustration(); // from illustrations.js
            caption = "图：Transformer 架构的简化示意图";
        }
        // Add more `else if` for other hardcoded SVGs if needed

        if (svgString) {
            const figure = document.createElement('figure');
            figure.className = 'svg-container my-6 anim-on-scroll'; // Added anim-on-scroll
            figure.dataset.animationType = "fadeInScaleUp";
            figure.innerHTML = svgString;

            const figcaptionElement = document.createElement('figcaption');
            figcaptionElement.className = 'text-center text-sm text-gray-600 mt-2';
            figcaptionElement.textContent = caption;
            figure.appendChild(figcaptionElement);
            
            p.parentNode.replaceChild(figure, p);
        }
    });

    // Process raw ```svg code blocks from Markdown
    container.querySelectorAll('pre code.language-svg').forEach((block, index) => {
        const svgContent = block.textContent;
        const preElement = block.parentNode; 

        if (preElement.classList.contains('code-block-processed-as-svg') || preElement.closest('.svg-container')) {
            return; // Already handled or part of an illustration.js SVG
        }
        
        const figure = document.createElement('figure');
        // Added markdown-injected-svg for specific styling from styles.css
        figure.className = 'svg-container my-6 anim-on-scroll markdown-injected-svg'; 
        figure.dataset.animationType = "fadeInScaleUp";
        figure.dataset.animationDelay = (index * 0.05).toString();
        
        // Sanitize SVG content slightly for safety if needed, but generally trust markdown source
        // For now, directly inserting:
        figure.innerHTML = svgContent; 
        
        let captionText = "SVG Illustration"; 
        const prevSibling = preElement.previousElementSibling;
        if (prevSibling && prevSibling.tagName === 'P' && prevSibling.textContent.toLowerCase().match(/^图[：:]/)) { // Matches "图：" or "图:"
            captionText = prevSibling.textContent;
            prevSibling.remove(); 
        } else {
            // Check if the SVG itself contains a <title> element to use as caption
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = svgContent;
            const svgTitle = tempDiv.querySelector('svg > title');
            if (svgTitle && svgTitle.textContent.trim()) {
                captionText = svgTitle.textContent.trim();
            }
        }


        const figcaptionElement = document.createElement('figcaption');
        figcaptionElement.className = 'text-center text-sm text-gray-600 mt-2';
        figcaptionElement.textContent = captionText;
        figure.appendChild(figcaptionElement);

        preElement.parentNode.replaceChild(figure, preElement);
        preElement.classList.add('code-block-processed-as-svg'); // Mark as processed
    });
}

