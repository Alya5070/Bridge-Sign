import os
import re

template_dir = r"d:\Python Project\Hand Sign Detection\templates"

nav_regex = re.compile(r'<nav\b[^>]*\bclass="navbar\b[^>]*>.*?</nav>\s*(?:<div\s+class="version-strip".*?</div>)?', re.DOTALL)
head_regex = re.compile(r'</head>', re.IGNORECASE)

link_tag = '    <link rel="stylesheet" href="{{ url_for(\'static\', filename=\'css/global.css\') }}">\n</head>'

for filename in os.listdir(template_dir):
    if filename.endswith(".html") and filename != "_navbar.html":
        filepath = os.path.join(template_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        changed = False
        
        # Inject CSS
        if 'global.css' not in content:
            content, count = head_regex.subn(link_tag, content)
            if count > 0: changed = True

        # Replace Nav
        if '{% include \'_navbar.html\' %}' not in content:
            content, count = nav_regex.subn('{% include \'_navbar.html\' %}', content)
            if count > 0: changed = True

        if changed:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated {filename}")
