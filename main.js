document.addEventListener('DOMContentLoaded', function() {
    initMobileMenu();
    renderHeroIllustration(); 
    generateTOC(); 

    const transformerBentoPlaceholder = document.getElementById('transformer-illustration-bento');
    if (transformerBentoPlaceholder) {
        transformerBentoPlaceholder.innerHTML = createTransformerIllustration(); 
    }

    const projectQuoteCardPlaceholder = document.getElementById('project-quote-card');
    if (projectQuoteCardPlaceholder) {
        // Example Quote
        const quoteCardElement = createQuoteCard(
            "The only way to do great work is to love what you do.",
            "Steve Jobs",
            "format_quote" 
        );
        // The placeholder div itself has anim-on-scroll and col-span
        projectQuoteCardPlaceholder.appendChild(quoteCardElement);
    }


    loadAndProcessAllContent().then(() => { 
        setupNumberCounters(); 
        highlightCodeBlocks(); 
        setupNavigationHighlighting(); 

        if (typeof initScrollAnimations === 'function') {
            initScrollAnimations(); 
        } else {
            console.warn('initScrollAnimations function not found. Animations might not work.');
        }
        
        console.log("All content loaded and processed. UI enhancements applied.");

    }).catch(error => {
        console.error("Fatal error during content loading and processing pipeline:", error);
        const body = document.querySelector('body');
        if (body) {
            body.innerHTML = `<div class="p-8 text-center text-red-600 bg-red-50">
                <h1>An Error Occurred</h1>
                <p>Sorry, the content could not be loaded. Please try again later.</p>
                <p><em>${error.message}</em></p>
            </div>`;
        }
    });

    setupReadingProgress(); 
    setupBackToTop(); 
});

function initMobileMenu() {
    const menuButton = document.getElementById('mobileMenuButton');
    const mobileMenu = document.getElementById('mobileMenu');
    if (!menuButton || !mobileMenu) return;

    menuButton.addEventListener('click', () => {
        mobileMenu.classList.toggle('hidden');
        menuButton.querySelector('.material-icons-outlined').textContent = mobileMenu.classList.contains('hidden') ? 'menu' : 'close';
    });

    document.querySelectorAll('#mobileMenu a.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            mobileMenu.classList.add('hidden');
            menuButton.querySelector('.material-icons-outlined').textContent = 'menu';
            // Smooth scroll for mobile links
            const targetId = link.getAttribute('href');
            if (targetId.startsWith('#')) {
                scrollToElement(targetId);
            }
        });
    });
}

function generateTOC() {
    const tocContainer = document.querySelector('.toc-grid');
    if (!tocContainer) return;

    const tocItems = [
        { id: 'intro-section', title: '1. AI Agent 导论', description: '基础概念、发展历程、应用领域与学习意义', icon: 'emoji_objects' },
        { id: 'algorithms-section', title: '2. 核心算法基础', description: '搜索、规划、机器学习、强化学习、NLP、LLM、知识表示', icon: 'psychology' },
        { id: 'technologies-section', title: '3. 关键技术栈与工具', description: '编程语言、AI/ML库、Agent框架、数据处理与部署工具', icon: 'developer_mode' },
        { id: 'projects-section', title: '4. 实践项目', description: '构建一个小型AI Agent的完整流程和代码实践', icon: 'fact_check' },
        { id: 'advanced-section', title: '5. 进阶学习与未来展望', description: '多智能体、可解释性、持续学习、挑战与机遇', icon: 'trending_up' },
        { id: 'conclusion-section', title: '6. 指南总结与最终建议', description: '知识体系回顾、学习资源与未来规划', icon: 'task_alt' }
    ];

    tocContainer.innerHTML = ''; 
    tocItems.forEach((item, index) => {
        const card = createBentoCard(item.title, item.description, item.id, item.icon);
        card.classList.add('anim-on-scroll');
        card.dataset.animationType = 'fadeInScaleUp'; 
        card.dataset.animationDelay = (index * 0.05).toString(); 
        tocContainer.appendChild(card);
    });
}

