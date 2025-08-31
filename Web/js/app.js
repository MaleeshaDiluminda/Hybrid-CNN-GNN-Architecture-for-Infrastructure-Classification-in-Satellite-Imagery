// Sidebar navigation controller
const sidebarButtons = document.querySelectorAll('.sidebar-btn');
const contentPanels = document.querySelectorAll('.content-panel');

function activatePanel(tabId) {
  // Update sidebar buttons
  sidebarButtons.forEach(btn => {
    const active = btn.dataset.tab === tabId;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  
  // Update content panels
  contentPanels.forEach(panel => {
    panel.classList.toggle('active', panel.id === tabId);
  });
  
  // Persist in URL hash
  history.replaceState(null, '', '#' + tabId);
}

// Add click event listeners to sidebar buttons
sidebarButtons.forEach(btn => {
  btn.addEventListener('click', () => activatePanel(btn.dataset.tab));
});

// Restore selected panel from URL hash
const initialTab = location.hash && location.hash.substring(1);
if (initialTab && document.getElementById(initialTab)) {
  activatePanel(initialTab);
}