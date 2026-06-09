// QFlow Website Main JavaScript

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add active state to navigation items
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section');
    const navItems = document.querySelectorAll('.navbar-item[href^="#"]');

    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (pageYOffset >= sectionTop - 60) {
            current = section.getAttribute('id');
        }
    });

    navItems.forEach(item => {
        item.classList.remove('is-active');
        if (item.getAttribute('href').slice(1) === current) {
            item.classList.add('is-active');
        }
    });
});
