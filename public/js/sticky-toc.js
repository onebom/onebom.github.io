document.addEventListener('DOMContentLoaded', function () {
    const stickyToc = document.getElementById('sticky-toc');

    if (!stickyToc) {
        return; // Don't run if the TOC isn't on the page
    }

    const initialTop = 180;
    const finalTop = 30;

    function adjustTocPosition() {
        const scrollY = window.scrollY;
        const newTop = Math.max(finalTop, initialTop - scrollY);
        stickyToc.style.top = newTop + 'px';
    }

    // Adjust position on initial load
    adjustTocPosition();

    // Adjust position on scroll
    window.addEventListener('scroll', adjustTocPosition, { passive: true });
});
