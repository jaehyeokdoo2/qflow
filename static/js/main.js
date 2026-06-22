// Dataset thumbnail hover → highlight matching rows in method cards
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.dataset-thumb').forEach(thumb => {
        thumb.addEventListener('mouseenter', () => {
            const ds = thumb.dataset.dataset;
            document.querySelectorAll('.dilemma-fig-row').forEach(row => {
                if (row.dataset.dataset === ds) {
                    row.classList.add('row-highlight');
                    row.classList.remove('row-dim');
                } else {
                    row.classList.add('row-dim');
                    row.classList.remove('row-highlight');
                }
            });
        });
        thumb.addEventListener('mouseleave', () => {
            document.querySelectorAll('.dilemma-fig-row').forEach(row => {
                row.classList.remove('row-highlight', 'row-dim');
            });
        });
    });
});

// Re-render MathJax when algo-toggle details are opened
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.algo-toggle').forEach(details => {
        details.addEventListener('toggle', () => {
            if (details.open && window.MathJax) {
                MathJax.typesetPromise([details]);
            }
        });
    });
});

// Collapsible results tables
document.querySelectorAll('.expand-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const table = btn.previousElementSibling;
        const hiddenRows = table.querySelectorAll('.hidden-row');
        const label = btn.querySelector('.expand-label');
        const icon = btn.querySelector('.expand-icon');
        const isExpanded = btn.dataset.expanded === 'true';

        hiddenRows.forEach(row => {
            row.style.display = isExpanded ? 'none' : '';
        });

        btn.dataset.expanded = isExpanded ? 'false' : 'true';
        label.textContent = isExpanded ? 'Show all environments' : 'Show fewer';
        icon.classList.toggle('fa-chevron-down', isExpanded);
        icon.classList.toggle('fa-chevron-up', !isExpanded);
    });
});


// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});
