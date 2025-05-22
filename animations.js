// Manages all scroll-triggered animations using Framer Motion

function initScrollAnimations() {
    // More robust check for Framer Motion
    if (typeof window.motion === 'undefined' || !window.motion) {
        console.warn('Framer Motion (motion) not found. Scroll animations will be basic or fallback to CSS if defined.');
        // Fallback: Add a class that CSS can use for simple fade-in
        document.querySelectorAll('.anim-on-scroll').forEach(el => {
            const observer = new IntersectionObserver(entries => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = '1'; // Simple opacity fallback
                        entry.target.style.transform = 'translateY(0)'; // Simple transform fallback
                        observer.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.1 });
            observer.observe(el);
        });
        return;
    }

    const elementsToAnimate = document.querySelectorAll('.anim-on-scroll');

    const observer = new IntersectionObserver(
        (entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const el = entry.target;
                    const animationType = el.dataset.animationType || 'fadeInUp';
                    const delay = parseFloat(el.dataset.animationDelay) || 0;
                    const duration = parseFloat(el.dataset.animationDuration) || 0.6; // Default duration 0.6s
                    const stagger = parseFloat(el.dataset.staggerChildren) || 0; // For parent elements

                    let animationProps = {
                        initial: { opacity: 0, y: 20 },
                        animate: { opacity: 1, y: 0 },
                    };

                    switch (animationType) {
                        case 'fadeIn':
                            animationProps = { initial: { opacity: 0 }, animate: { opacity: 1 } };
                            break;
                        case 'fadeInUp': // Default
                            animationProps = { initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0 } };
                            break;
                        case 'fadeInDown':
                            animationProps = { initial: { opacity: 0, y: -20 }, animate: { opacity: 1, y: 0 } };
                            break;
                        case 'fadeInLeft':
                            animationProps = { initial: { opacity: 0, x: 20 }, animate: { opacity: 1, x: 0 } };
                            break;
                        case 'fadeInRight':
                            animationProps = { initial: { opacity: 0, x: -20 }, animate: { opacity: 1, x: 0 } };
                            break;
                        case 'fadeInScaleUp':
                            animationProps = { initial: { opacity: 0, scale: 0.95, y:10 }, animate: { opacity: 1, scale: 1, y:0 } };
                            break;
                    }
                    
                    // Apply initial styles directly before animating to ensure they are set
                    Object.keys(animationProps.initial).forEach(prop => {
                        el.style[prop] = animationProps.initial[prop] + (typeof animationProps.initial[prop] === 'number' && prop !== 'opacity' && prop !== 'scale' ? 'px' : '');
                    });


                    motion.animate(
                        el,
                        animationProps.animate,
                        { 
                            duration: duration, 
                            delay: delay, 
                            ease: [0.16, 1, 0.3, 1] // Smooth easing (Quintic Out)
                        }
                    );

                    if (stagger > 0) {
                        const children = el.querySelectorAll('.anim-child');
                        children.forEach((child, index) => {
                            // Similar animation for children, but with staggered delay
                             Object.keys(animationProps.initial).forEach(prop => {
                                child.style[prop] = animationProps.initial[prop] + (typeof animationProps.initial[prop] === 'number' && prop !== 'opacity' && prop !== 'scale' ? 'px' : '');
                            });
                            motion.animate(
                                child,
                                animationProps.animate,
                                { 
                                    duration: duration, 
                                    delay: delay + (index * stagger), 
                                    ease: [0.16, 1, 0.3, 1] 
                                }
                            );
                        });
                    }

                    obs.unobserve(el); // Animate only once
                }
            });
        },
        {
            threshold: 0.1, // Trigger when 10% of the element is visible
            rootMargin: "0px 0px -50px 0px" // Start animation a bit before it's fully in view from bottom
        }
    );

    elementsToAnimate.forEach(el => {
        observer.observe(el);
    });
}

// Note: The original content of animations.js related to DOMContentLoaded and setTimeout for new elements
// is removed as initScrollAnimations will be called explicitly after all content is loaded in main.js.
// Number counter animations are handled in utils.js.


