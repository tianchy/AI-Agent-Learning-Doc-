// Utility functions

async function fetchTextContent(url) {
    console.log(`Fetching content from: ${url}`);
    try {
        const response = await fetch(url);
        console.log(`Response status: ${response.status}`);
        
        if (!response.ok) {
            if (response.status === 404) {
                console.error(`Content file not found: ${url}`);
                return `## Content Not Found\n\nThe file \`${url}\` could not be loaded. Please check the path.`;
            }
            throw new Error(`HTTP error! status: ${response.status} for ${url}`);
        }
        
        const text = await response.text();
        console.log(`Content length: ${text.length} characters`);
        console.log(`Content preview: ${text.substring(0, 100)}...`);
        
        if (!text || typeof text !== 'string') {
            console.error(`Invalid content from ${url}:`, text);
            return `## Error Loading Content\n\nInvalid content format from \`${url}\`.`;
        }
        
        return text;
    } catch (error) {
        console.error(`Error fetching ${url}:`, error);
        return `## Error Loading Content\n\nThere was an issue fetching content from \`${url}\`.\n\n\`\`\`\n${error.message}\n\`\`\``;
    }
}

function parseMarkdown(text) {
    if (typeof marked === 'undefined') {
        console.error('Marked library is not loaded');
        return { metadata: {}, content: '<p class="text-red-500">Error: Markdown parser (marked) is not loaded.</p>' };
    }
    
    if (!text || typeof text !== 'string') {
        console.error('Invalid input to parseMarkdown:', text);
        return { metadata: {}, content: '<p class="text-red-500">Error: Invalid input to markdown parser.</p>' };
    }

    try {
        // 确保marked库已正确初始化
        if (typeof marked.parse !== 'function') {
            console.error('Marked library is not properly initialized');
            return { metadata: {}, content: '<p class="text-red-500">Error: Markdown parser is not properly initialized.</p>' };
        }

        let markdownContent = text;
        if (text.startsWith('---')) {
            const parts = text.split('---');
            if (parts.length >= 3) {
                markdownContent = parts.slice(2).join('---').trim();
            }
        } else {
            markdownContent = text.trim();
        }

        let preProcessedContent = markdownContent;
        preProcessedContent = processFeatureBlocks(preProcessedContent); 
        preProcessedContent = processStatsBlocks(preProcessedContent);   

        const renderer = new marked.Renderer();

        renderer.paragraph = function(text) {
            if (!text) return '<p></p>\n';
            const processedText = text.replace(/`([^`]+)`/g, '<code class="font-english">$1</code>');
            return `<p>${processedText}</p>\n`;
        };
        
        renderer.listitem = function(text) {
            if (!text) return '<li></li>\n';
            const processedText = text.replace(/`([^`]+)`/g, '<code class="font-english">$1</code>');
            return `<li>${processedText}</li>\n`;
        };

        renderer.heading = function (text, level, raw) {
            if (!text) return '';
            const id = text.toLowerCase().replace(/[^\w]+/g, '-');
            return `<h${level} id="${id}" class="font-chinese">${text}</h${level}>\n`;
        };

        renderer.table = function(header, body) {
            return `<div class="overflow-x-auto"><table class="markdown-table">${header}${body}</table></div>\n`;
        };

        renderer.blockquote = function(quote) {
            return `<blockquote>${quote}</blockquote>\n`;
        }

        // 设置marked选项
        marked.setOptions({
            gfm: true,
            breaks: true, 
            pedantic: false,
            sanitize: false, 
            smartLists: true,
            smartypants: false,
            langPrefix: 'language-',
            renderer: renderer
        });

        // 使用try-catch包装marked.parse调用
        let htmlContent;
        try {
            console.log('Parsing markdown content...');
            console.log('Content preview:', preProcessedContent.substring(0, 100));
            
            htmlContent = marked.parse(preProcessedContent);
            
            if (!htmlContent || typeof htmlContent !== 'string') {
                throw new Error('Marked parser returned invalid output');
            }
            
            console.log('Markdown parsing successful');
            console.log('HTML preview:', htmlContent.substring(0, 100));
            
        } catch (parseError) {
            console.error('Error in marked.parse:', parseError);
            return { metadata: {}, content: `<p class="text-red-500">Error parsing markdown content: ${parseError.message}</p>` };
        }

        return { metadata: {}, content: htmlContent };
    } catch (error) {
        console.error('Error parsing markdown:', error);
        return { metadata: {}, content: `<p class="text-red-500">Error parsing content: ${error.message}</p>` };
    }
}

function processStatsBlocks(text) {
    const statsPattern = /\[STATS\]([\s\S]*?)\[\/STATS\]/g;
    return text.replace(statsPattern, (match, statsContent) => {
        const statItems = statsContent
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.startsWith('-'))
            .map(line => {
                const parts = line.substring(1).trim().split(':');
                if (parts.length < 2) return null;

                const rawValueAndSuffix = parts[0].trim();
                const labelAndHint = parts.slice(1).join(':').trim();
                
                let label = labelAndHint;
                let iconHint = null; // Not used by createStatsCounter currently

                // Regex to separate number from suffix part
                const valueMatch = rawValueAndSuffix.match(/^([\d\.]+)\s*(.*)$/);
                let value, suffix = '';

                if (valueMatch) {
                    value = parseFloat(valueMatch[1]);
                    suffix = (valueMatch[2] || '').trim();
                } else {
                    value = parseFloat(rawValueAndSuffix); // Try parsing as plain number
                }
                
                if (isNaN(value)) {
                     console.warn("Could not parse stat value:", rawValueAndSuffix);
                     return null;
                }
                // If suffix from value part is empty, try to get it from label part (e.g., "预算_十亿美元")
                if (!suffix && labelAndHint.includes('_')) {
                    const labelParts = labelAndHint.split('_');
                    label = labelParts[0].trim();
                    const hintSuffix = labelParts[1].trim();
                    if (hintSuffix === '%' || hintSuffix.toLowerCase().includes('usd') || hintSuffix.toLowerCase().includes('b') || hintSuffix === '/7') {
                        suffix = hintSuffix;
                    }
                } else if (labelAndHint.includes('_')) {
                     label = labelAndHint.split('_')[0].trim();
                }


                return { value, suffix, label };
            })
            .filter(item => item !== null);

        if (statItems.length === 0) return match;
        const tempDiv = document.createElement('div');
        const statsCounterElement = createStatsCounter(statItems); // from components.js
        statsCounterElement.classList.add('anim-on-scroll');
        statsCounterElement.dataset.animationType = 'fadeInUp'; 
        tempDiv.appendChild(statsCounterElement);
        return tempDiv.innerHTML;
    });
}

function processFeatureBlocks(text) {
    const featurePattern = /\[FEATURES:\s*(.*?)\]([\s\S]*?)\[\/FEATURES\]/g;
    return text.replace(featurePattern, (match, title, featuresContent) => {
        const features = featuresContent
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.startsWith('-'))
            .map(line => {
                const parts = line.substring(1).trim().split('|');
                if (parts.length < 2) return null; 
                return {
                    title: (parts[0] || '').trim(),
                    description: (parts[1] || '').trim(),
                    icon: (parts[2] || 'check_circle_outline').trim() 
                };
            })
            .filter(item => item !== null);

        if (features.length === 0) return match;
        
        let featureListHTML = createFeatureList(title.trim(), features); // from components.js
        
        // Ensure anim-on-scroll is added to the root of the generated HTML
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = featureListHTML;
        const featureListContainer = tempDiv.firstChild;
        if (featureListContainer && !featureListContainer.classList.contains('anim-on-scroll')) {
            featureListContainer.classList.add('anim-on-scroll');
            featureListContainer.dataset.animationType = 'fadeInUp';
            featureListContainer.dataset.animationDelay = '0.1'; // Default delay, can be staggered if needed
        }
        return tempDiv.innerHTML;
    });
}


function highlightCodeBlocks() {
    document.querySelectorAll('.markdown-content pre code[class*="language-"]').forEach((block, index) => {
        const preElement = block.parentNode; 
        if (preElement.closest('.code-block-container') || preElement.classList.contains('code-block-processed')) return;

        const languageMatch = block.className.match(/language-(\w+)/);
        const language = languageMatch ? languageMatch[1] : 'code';

        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-container anim-on-scroll';
        wrapper.dataset.animationType = 'fadeInUp';
        wrapper.dataset.animationDelay = (index * 0.03).toString(); // Slightly faster stagger

        const header = document.createElement('div');
        header.className = 'code-block-header';

        const langLabel = document.createElement('span');
        langLabel.className = 'language-label';
        langLabel.textContent = language.toUpperCase();

        const copyButton = document.createElement('button');
        copyButton.className = 'copy-code-button';
        copyButton.innerHTML = `<span class="material-icons-outlined">content_copy</span> Copy`;
        copyButton.setAttribute('aria-label', 'Copy code to clipboard');

        copyButton.addEventListener('click', () => {
            navigator.clipboard.writeText(block.textContent).then(() => {
                copyButton.innerHTML = `<span class="material-icons-outlined">check</span> Copied!`;
                setTimeout(() => {
                    copyButton.innerHTML = `<span class="material-icons-outlined">content_copy</span> Copy`;
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy code: ', err);
                copyButton.textContent = 'Error';
                 setTimeout(() => {
                    copyButton.innerHTML = `<span class="material-icons-outlined">content_copy</span> Copy`;
                }, 2000);
            });
        });

        header.appendChild(langLabel);
        header.appendChild(copyButton);
        
        preElement.parentNode.insertBefore(wrapper, preElement);
        wrapper.appendChild(header);
        wrapper.appendChild(preElement); 

        preElement.classList.add('code-block-content'); 
        preElement.classList.add('code-block-processed');
        preElement.classList.remove('anim-on-scroll'); // Wrapper handles animation
        preElement.removeAttribute('data-animation-type');
        preElement.removeAttribute('data-animation-delay');


        if (window.Prism) {
            Prism.highlightElement(block);
        } else {
            console.warn('Prism.js not loaded, cannot highlight code blocks.');
        }
    });
}


function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function scrollToElement(elementId, offset = 90) { 
    const id = elementId.startsWith('#') ? elementId.substring(1) : elementId;
    const element = document.getElementById(id);
    if (element) {
        const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
        const offsetPosition = elementPosition - offset;

        window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
        });
    } else {
        console.warn(`Element with ID "${id}" not found for scrolling.`);
    }
}


function animateNumber(element, start, end, duration) {
    let startTimestamp = null;
    const step = timestamp => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = Math.floor(progress * (end - start) + start);
        element.innerText = value.toLocaleString(); 

        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
             element.innerText = end.toLocaleString(); 
        }
    };
    window.requestAnimationFrame(step);
}

function setupNumberCounters() {
    const countElements = document.querySelectorAll('.count-up:not(.animated)');

    const countObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const target = entry.target;
                    const finalValue = parseFloat(target.getAttribute('data-value'));
                    if (isNaN(finalValue)) {
                        console.warn("Invalid data-value for count-up:", target.getAttribute('data-value'));
                        target.innerText = target.getAttribute('data-value'); // Display original if not a number
                        target.classList.add('animated'); 
                        countObserver.unobserve(target);
                        return;
                    }
                    target.innerText = "0"; 
                    animateNumber(target, 0, finalValue, 1500); 
                    target.classList.add('animated'); 
                    countObserver.unobserve(target); 
                }
            });
        },
        {
            threshold: 0.5 
        }
    );

    countElements.forEach(element => {
        countObserver.observe(element);
    });
}


function setupNavigationHighlighting() {
    const sections = document.querySelectorAll('section[id$="-section"]'); 
    const navLinks = document.querySelectorAll('header nav .nav-link'); // More specific selector for desktop nav
    const mobileNavLinks = document.querySelectorAll('#mobileMenu .nav-link');

    if (!sections.length || (!navLinks.length && !mobileNavLinks.length)) {
        console.warn("Navigation highlighting skipped: Sections or nav links not found.");
        return;
    }

    const headerHeight = document.querySelector('header')?.offsetHeight || 80;

    const updateActiveLink = () => {
        let currentSectionId = '';
        // Iterate from bottom to top to find the current section
        // This helps with sections that are shorter than the viewport or when at the bottom of the page
        for (let i = sections.length - 1; i >= 0; i--) {
            const section = sections[i];
            const rect = section.getBoundingClientRect();
             // Check if the top of the section is within a certain range from the top of the viewport
             // Or if the section is mostly visible
            if (rect.top <= headerHeight + 100 && rect.bottom >= headerHeight + 100) {
                 currentSectionId = section.getAttribute('id');
                 break;
            }
        }
        
        // Fallback if no section is "current" (e.g., scrolled to very top or bottom beyond sections)
        if (!currentSectionId && sections.length > 0) {
            if (window.scrollY < sections[0].offsetTop - headerHeight) { // Above the first section
                // currentSectionId = sections[0].getAttribute('id'); // Or no active link
            } else if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 50) { // At the very bottom
                currentSectionId = sections[sections.length - 1].getAttribute('id');
            } else { // Default to first visible if logic above fails
                 for (const section of sections) {
                    if (section.offsetTop <= window.scrollY + headerHeight + 50) {
                        currentSectionId = section.getAttribute('id');
                    } else {
                        break; 
                    }
                }
            }
        }


        [...navLinks, ...mobileNavLinks].forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSectionId}`) {
                link.classList.add('active');
            }
        });
    };

    const debouncedUpdate = debounce(updateActiveLink, 50); 
    window.addEventListener('scroll', debouncedUpdate);
    window.addEventListener('resize', debouncedUpdate);
    updateActiveLink(); // Initial call
}

function setupReadingProgress() {
    const progressBar = document.getElementById('readingProgress');
    if (!progressBar) return;

    const updateProgress = () => {
        const scrollPosition = window.scrollY;
         const totalHeight = document.documentElement.scrollHeight - window.innerHeight;

        if (totalHeight <= 0) {
            progressBar.style.transform = 'scaleX(0)'; 
            return;
        }
        const progress = Math.min(1, Math.max(0, scrollPosition / totalHeight));
        progressBar.style.transform = `scaleX(${progress})`;
    };

    const debouncedUpdateProgress = debounce(updateProgress, 10); 
    window.addEventListener('scroll', debouncedUpdateProgress);
    window.addEventListener('resize', debouncedUpdateProgress);
    updateProgress(); 
}

function setupBackToTop() {
    const backToTopButton = document.getElementById('backToTop');
    if (!backToTopButton) return;

    const toggleButton = () => {
        if (window.scrollY > 300) {
            backToTopButton.classList.remove('opacity-0', 'invisible');
            backToTopButton.classList.add('opacity-100', 'visible');
        } else {
            backToTopButton.classList.remove('opacity-100', 'visible');
            backToTopButton.classList.add('opacity-0', 'invisible');
        }
    };

    backToTopButton.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    window.addEventListener('scroll', debounce(toggleButton, 100));
    toggleButton(); 
}

