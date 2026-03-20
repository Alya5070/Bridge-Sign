import os
import re

directory = r'D:\Python Project\Hand Sign Detection\templates'

# Unified Navbar template using request.endpoint to highlight the active tab dynamically
navbar_template = '''    <nav class="navbar glass-panel">
        <div class="navbar-brand">Bridge Sign</div>
        <div class="navbar-menu">
            <a href="{{ url_for('index') }}" class="navbar-item{% if request.endpoint == 'index' %} active{% endif %}">Dashboard</a>
            <a href="{{ url_for('tutorial') }}" class="navbar-item{% if request.endpoint == 'tutorial' %} active{% endif %}">Tutorial</a>
            <a href="{{ url_for('quiz') }}" class="navbar-item{% if request.endpoint == 'quiz' %} active{% endif %}">Quiz</a>
            <a href="{{ url_for('profile') }}" class="navbar-item{% if request.endpoint == 'profile' %} active{% endif %}">Profile</a>
            <a href="{{ url_for('settings') }}" class="navbar-item{% if request.endpoint == 'settings' %} active{% endif %}">Settings</a>
            {% if is_admin %}
            <a href="{{ url_for('trainer') }}" class="navbar-item{% if request.endpoint == 'trainer' %} active{% endif %}">Trainer Module</a>
            <a href="{{ url_for('admin_dashboard') }}" class="navbar-item{% if request.endpoint == 'admin_dashboard' %} active{% endif %}">Manage Users</a>
            {% endif %}
        </div>
        <div class="navbar-user">
            <div class="user-avatar">{{ username[0].upper() if username else 'U' }}</div>
            <span>{{ username }}</span>
            <a href="{{ url_for('logout') }}" class="logout-btn">Logout</a>
        </div>
    </nav>'''

for filename in os.listdir(directory):
    if filename.endswith('.html'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Regex match the entire <nav> block and replace it
        new_content = re.sub(r'<nav class="navbar([\s\S]*?)</nav>', navbar_template, content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

print("Navbar globally unified and ghost duplicates purged!")
