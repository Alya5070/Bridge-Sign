import os
import re

templates_dir = r"D:\Python Project\Hand Sign Detection\templates"
files_to_patch = ['admin_dashboard.html', 'dashboard.html', 'profile.html', 'quiz.html', 'settings.html', 'trainer.html', 'tutorial.html']

for filename in files_to_patch:
    filepath = os.path.join(templates_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Navbar text replacement
    content = content.replace('Trainer</a>', 'Trainer Module</a>')
    content = content.replace('Admin</a>', 'Manage Users</a>')
    
    # CSS Navbar adjustments (prevent wrapping)
    # Target .navbar
    content = re.sub(r'(\.navbar\s*\{[^\}]*)', r'\1\n        flex-wrap: nowrap;\n        overflow-x: auto;', content)
    # Target .navbar-item
    content = re.sub(r'(\.navbar-item\s*\{[^\}]*)', r'\1\n        white-space: nowrap;', content)

    # Specific file patches
    if filename == 'admin_dashboard.html':
        content = content.replace('Admin Dashboard</h1>', 'Manage Users</h1>')
        content = content.replace('<p>Manage spelling model versions and backups</p>', '<p>Manage access privileges to the platform.</p>')
        
        # Remove the Publish and Rollback cards
        content = re.sub(
            r'<div class="admin-card">[\s\S]*?Publish New Model[\s\S]*?</form>\s*</div>\s*<div class="admin-card">[\s\S]*?Rollback to Previous Version[\s\S]*?</form>\s*</div>', 
            '', 
            content
        )

    if filename == 'trainer.html':
        # Add labels readout
        old_action_btns = '''<div class="action-buttons">
                <button class="btn-admin btn-train" id="trainBtn" onclick="trainModel()">Train New Model</button>
                <button class="btn-admin btn-test" id="testBtn" onclick="toggleTesting()">Test Trained Model</button>
            </div>'''
        new_action_btns = old_action_btns + '''\n            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 12px;">
                <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.5rem;"><strong>Labels in the system:</strong></p>
                <p id="currentLabelsText" style="color: #fff; font-weight: 500; font-size: 1rem; word-wrap: break-word;">Loading...</p>
            </div>'''
        content = content.replace(old_action_btns, new_action_btns)

        # JS Fetch Script injection
        fetch_js = '''let trainingInterval;\n
        function fetchSystemLabels() {
            fetch('/get_dataset_labels')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('currentLabelsText').innerText = data.labels.join(', ') || 'No labels recorded yet.';
                    } else {
                        document.getElementById('currentLabelsText').innerText = 'Error loading labels.';
                    }
                });
        }
        window.addEventListener('DOMContentLoaded', fetchSystemLabels);'''
        content = content.replace('let trainingInterval;', fetch_js)
        
        # Trigger updates on save/delete
        content = content.replace(
            'term.innerHTML += `> <span style="color:#ef4444">${data.message}</span><br>`;',
            'term.innerHTML += `> <span style="color:#ef4444">${data.message}</span><br>`;\n                    if(data.success) fetchSystemLabels();'
        )
        content = content.replace(
            'document.getElementById(\'recordBtn\').innerText = "Data Saved! Record again if needed";',
            'document.getElementById(\'recordBtn\').innerText = "Data Saved! Record again if needed";\n                    fetchSystemLabels();'
        )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("Recovery and patching completed successfully!")
