// Global variables
let currentChapterId = 'chapter1'; // Store ID instead of full object
const chaptersConfig = [
    { id: 'chapter1', title: '1. AI Agent 导论', file: 'chapters/chapter1.md' },
    { id: 'chapter2', title: '2. 核心算法基础', file: 'chapters/chapter2.md' },
    { id: 'chapter3', title: '3. 关键技术栈与工具', file: 'chapters/chapter3.md' },
    { id: 'chapter4', title: '4. 实践项目：构建一个小型 AI Agent', file: 'chapters/chapter4.md' },
    { id: 'chapter5', title: '5. 进阶学习与未来展望', file: 'chapters/chapter5.md' },
    { id: 'chapter6', title: '6. 指南总结与最终建议', file: 'chapters/chapter6.md' }
];

// DOM Elements
let tocElement, contentContainer, mobileMenu, searchContainer, sidebar, sidebarToggle, sidebarOverlay, scrollToTopBtn;

document.addEventListener('DOMContentLoaded', () => {
    tocElement = document.getElementById('toc');
    contentContainer = document.getElementById('contentContainer');
    mobileMenu = document.getElementById('mobileMenu');
    searchContainer = document.getElementById('searchContainer');
    sidebar = document.getElementById('sidebar');
    sidebarToggle = document.getElementById('sidebarToggle');
    sidebarOverlay = document.getElementById('sidebarOverlay');
    scrollToTopBtn = document.getElementById('scrollToTopBtn');
    
    if (!tocElement || !contentContainer) {
        console.error("Essential elements (TOC or ContentContainer) not found!");
        return;
    }

    buildTableOfContents();
    
    // Load chapter based on hash or default
    const chapterIdFromHash = window.location.hash.substring(1);
    loadChapter(chaptersConfig.find(ch => ch.id === chapterIdFromHash) ? chapterIdFromHash : currentChapterId);

    // Event Listeners
    window.addEventListener('hashchange', () => {
        const chapterIdFromHash = window.location.hash.substring(1);
        if (chapterIdFromHash && chapterIdFromHash !== currentChapterId) {
            loadChapter(chapterIdFromHash);
        }
    });

    if (scrollToTopBtn) {
        scrollToTopBtn.addEventListener('click', scrollToTop);
        window.addEventListener('scroll', handleScrollButtonVisibility);
    }
});

function buildTableOfContents() {
    tocElement.innerHTML = ''; // Clear existing TOC
    chaptersConfig.forEach(chapter => {
        const chapterLink = document.createElement('a');
        chapterLink.className = 'toc-link';
        chapterLink.href = `#${chapter.id}`;
        chapterLink.textContent = chapter.title;
        chapterLink.onclick = (e) => {
            e.preventDefault();
            if (chapter.id !== currentChapterId) {
                loadChapter(chapter.id);
            }
            if (window.innerWidth < 768 && sidebar.classList.contains('active')) { // Close sidebar on mobile after click
                toggleSidebar();
            }
            // Manually set hash, because loadChapter might not if chapter is same
            window.location.hash = chapter.id;
        };
        tocElement.appendChild(chapterLink);
    });
}

async function loadChapter(chapterId) {
    currentChapterId = chapterId;
    const chapter = chaptersConfig.find(ch => ch.id === chapterId);
    
    if (!chapter) {
        console.error(`Chapter ${chapterId} not found in configuration.`);
        contentContainer.innerHTML = `<div class="p-4 bg-red-100 text-red-700 rounded-md">Error: Chapter content not found.</div>`;
        return;
    }
    
    // Update active state in TOC
    document.querySelectorAll('#toc .toc-link').forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === `#${chapterId}`);
    });
    
    // Update window hash and title
    window.location.hash = chapterId;
    document.title = `${chapter.title} - AI Agent 开发综合指南`;

    // Show loading state (optional, if skeleton isn't enough)
    // contentContainer.innerHTML = `<div class="animate-pulse">...</div>`; 
    
    try {
        const response = await fetch(chapter.file);
        if (!response.ok) throw new Error(`Failed to load chapter: ${response.statusText}`);
        
        const markdownText = await response.text();
        
        const parts = markdownText.split('---');
        let metadata = {};
        let content = markdownText;
        
        if (parts.length >= 3 && parts[0].trim() === '') {
            try {
                metadata = jsyaml.load(parts[1]);
                content = parts.slice(2).join('---').trim();
            } catch (e) {
                console.warn('Could not parse YAML front matter:', e);
            }
        }
        
        contentContainer.innerHTML = marked.parse(content);
        
        // Post-processing: fix images, highlight code
        postProcessRenderedContent(contentContainer);
        
        // Scroll to top of content area, not whole page, to keep header visible
        contentContainer.scrollTop = 0; // If content container itself scrolls
        document.documentElement.scrollTop = document.getElementById('content').offsetTop - document.querySelector('header').offsetHeight - 20;


    } catch (error) {
        console.error('Error loading chapter:', chapterId, error);
        contentContainer.innerHTML = `
            <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-md relative" role="alert">
                <strong class="font-bold">加载失败!</strong>
                <span class="block sm:inline">无法加载章节 "${chapter.title}". 请检查网络连接或稍后再试.</span>
            </div>
        `;
    }
}

function postProcessRenderedContent(container) {
    if (typeof fixImagePaths === 'function') {
        fixImagePaths(container);
    }
    if (typeof highlightCodeBlocks === 'function') {
        highlightCodeBlocks(container); // This can ensure Prism has the right classes
    }
    if (window.Prism) {
        Prism.highlightAllUnder(container); // Tell Prism to highlight new content
    }
}

function toggleMobileMenu() {
    mobileMenu.classList.toggle('hidden');
    const isExpanded = !mobileMenu.classList.contains('hidden');
    document.getElementById('mobileMenuButton').setAttribute('aria-expanded', isExpanded.toString());
}

function toggleSidebar() {
    sidebar.classList.toggle('active');
    sidebarOverlay.classList.toggle('hidden'); // Toggle overlay visibility
    sidebarToggle.classList.toggle('hidden-important'); // Hide burger when sidebar is open

    const isExpanded = sidebar.classList.contains('active');
    sidebarToggle.setAttribute('aria-expanded', isExpanded.toString());

    if (isExpanded) {
        document.body.style.overflow = 'hidden'; // Prevent body scroll when mobile sidebar is open
    } else {
        document.body.style.overflow = '';
    }
}

function toggleSearch() {
    searchContainer.classList.toggle('hidden');
    if (!searchContainer.classList.contains('hidden')) {
        document.getElementById('searchInput').focus();
        // If search is part of header, ensure mobile menu closes if open, or vice versa.
        if(!mobileMenu.classList.contains('hidden')) {
           toggleMobileMenu();
        }
    }
}

function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
    // If on a chapter page, reset hash to avoid confusion, or go to first chapter.
    // window.location.hash = chaptersConfig[0].id; 
    // loadChapter(chaptersConfig[0].id); // Optional: navigate to first chapter on "Home"
}

function handleScrollButtonVisibility() {
    if (window.scrollY > 200) {
        scrollToTopBtn.classList.add('opacity-100', 'translate-y-0');
        scrollToTopBtn.classList.remove('opacity-0', 'translate-y-4');
    } else {
        scrollToTopBtn.classList.remove('opacity-100', 'translate-y-0');
        scrollToTopBtn.classList.add('opacity-0', 'translate-y-4');
    }
}
