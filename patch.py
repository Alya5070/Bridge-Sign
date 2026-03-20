import os

directory = r'D:\Python Project\Hand Sign Detection\templates'
for filename in os.listdir(directory):
    if filename.endswith('.html'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Inject Trainer Module if missing
        if '{% if is_admin %}' in content and 'url_for(\'trainer\')' not in content:
            content = content.replace(
                '{% if is_admin %}',
                '{% if is_admin %}\n            <a href="{{ url_for(\'trainer\') }}" class="navbar-item">Trainer Module</a>'
            )
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        elif '{% if is_admin %}' not in content and '<div class="navbar-menu">' in content:
            # If for some reason the admin block was entirely deleted from a template
            block = '''
            {% if is_admin %}
            <a href="{{ url_for('trainer') }}" class="navbar-item">Trainer Module</a>
            <a href="{{ url_for('admin_dashboard') }}" class="navbar-item">Manage Users</a>
            {% endif %}'''
            content = content.replace('<div class="navbar-menu">', '<div class="navbar-menu">' + block)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

print("Template links explicitly applied!")
