// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Dark mode toggle functionality
    initDarkMode();
    
    // Mobile menu toggle
    initMobileMenu();
});

// Initialize dark mode functionality
function initDarkMode() {
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const htmlElement = document.documentElement;
    
    // Check for user preference in local storage
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    
    // Set initial mode based on user preference or system preference
    if (isDarkMode || (localStorage.getItem('darkMode') === null && 
        window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        htmlElement.classList.add('dark');
    } else {
        htmlElement.classList.remove('dark');
    }
    
    // Toggle dark mode on button click
    darkModeToggle.addEventListener('click', () => {
        htmlElement.classList.toggle('dark');
        
        // Save preference to local storage
        localStorage.setItem('darkMode', htmlElement.classList.contains('dark'));
    });
    
    // Listen for system preference changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
        if (localStorage.getItem('darkMode') === null) {
            htmlElement.classList.toggle('dark', event.matches);
        }
    });
}

// Initialize mobile menu functionality
function initMobileMenu() {
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    
    // Toggle mobile menu visibility
    mobileMenuButton.addEventListener('click', () => {
        mobileMenu.classList.toggle('hidden');
    });
    
    // Close mobile menu when clicking on a link
    const mobileMenuLinks = mobileMenu.querySelectorAll('a');
    mobileMenuLinks.forEach(link => {
        link.addEventListener('click', () => {
            mobileMenu.classList.add('hidden');
        });
    });
    
    // Close mobile menu when screen size changes to desktop
    window.addEventListener('resize', () => {
        if (window.innerWidth >= 768) { // md breakpoint in Tailwind
            mobileMenu.classList.add('hidden');
        }
    });
}

// Add scroll event for navbar styling
window.addEventListener('scroll', () => {
    const nav = document.querySelector('nav');
    
    if (window.scrollY > 50) {
        nav.classList.add('shadow-md', 'bg-opacity-90', 'backdrop-blur-sm');
    } else {
        nav.classList.remove('shadow-md', 'bg-opacity-90', 'backdrop-blur-sm');
    }
});

// Animate elements when they come into view
document.addEventListener('DOMContentLoaded', () => {
    const observerOptions = {
        root: null, // relative to viewport
        rootMargin: '0px',
        threshold: 0.1 // trigger when 10% of the element is visible
    };
    
    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe project cards
    document.querySelectorAll('#projects .grid > div').forEach(card => {
        observer.observe(card);
    });
    
    // Observe skill categories
    document.querySelectorAll('#skills .grid > div').forEach(skillGroup => {
        observer.observe(skillGroup);
    });
}); 