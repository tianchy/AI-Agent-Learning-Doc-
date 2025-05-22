// Search functionality (placeholder)
document.getElementById('searchInput').addEventListener('input', function(e) {
    const query = e.target.value.toLowerCase().trim();
    const searchResultsContainer = document.getElementById('searchResults');
    
    if (query.length < 2) {
        searchResultsContainer.innerHTML = '';
        return;
    }
    
    // Basic placeholder for search. For a real app, this would involve
    // fetching/indexing content and performing a search.
    searchResultsContainer.innerHTML = `
        <div class="p-3 text-sm text-slate-600 bg-slate-50 rounded-md">
            <p>搜索功能正在开发中...</p>
            <p>当前搜索: "${query}"</p>
            <p class="mt-2 text-xs">提示: 实际搜索会遍历所有章节内容。</p>
        </div>
    `;
});

// Handle image paths: Ensure they are relative to an `images` folder if not absolute
function fixImagePaths(container) {
    const images = container.querySelectorAll('img');
    images.forEach(img => {
        const src = img.getAttribute('src');
        if (src && !src.startsWith('http') && !src.startsWith('/') && !src.startsWith('images/')) {
            // Check if it might be an SVG referenced directly like "agent-cycle.svg"
            // and assume it's in the "images" folder.
            if (src.endsWith('.svg') || src.endsWith('.png') || src.endsWith('.jpg') || src.endsWith('.jpeg') || src.endsWith('.gif')) {
                 img.setAttribute('src', 'images/' + src);
            }
        }
    });
}

// Pre-process code blocks if needed before Prism, or add classes for Prism.
function highlightCodeBlocks(container) {
    // Marked.js usually adds language-xxxx class to <code> elements inside <pre>.
    // Prism.js will pick these up.
    // This function can be extended if specific pre-processing is needed.
    // For example, ensure all <pre><code> blocks have a language class for Prism's autoloader fallback.
    const codeBlocks = container.querySelectorAll('pre > code');
    codeBlocks.forEach(block => {
        if (!block.className.includes('language-')) {
            // Attempt to infer or set a default language if none is specified
            // For now, Prism's autoloader will attempt to guess or can use a default.
            // block.classList.add('language-plain'); // Example default
        }
    });
    // Note: Prism.highlightAllUnder(container) in navigation.js does the actual highlighting.
}

// Initial setup on DOMContentLoaded for elements not tied to chapter loading
document.addEventListener('DOMContentLoaded', () => {
    // Any general initializations for content.js can go here.
    // For example, if there were static interactive elements on the page not part of chapters.
});
