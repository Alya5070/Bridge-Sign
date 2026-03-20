import os

filepath = r'D:\Python Project\Hand Sign Detection\app2.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix global labels in load_custom_model
target_func = """def load_custom_model():
    global custom_model, custom_labels"""
replacement_func = """def load_custom_model():
    global custom_model, custom_labels, labels"""

if target_func in content:
    content = content.replace(target_func, replacement_func)

target_assign = """custom_labels = [line.strip() for line in f.readlines()]
        return True"""
replacement_assign = """custom_labels = [line.strip() for line in f.readlines()]
            labels = custom_labels
        return True"""

if target_assign in content:
    content = content.replace(target_assign, replacement_assign)

# 2. Add GeneratorExit fallback to all yields within generate_frames
target_yield1 = """                yield (b'--frame\\r\\n'
                       b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')"""
replacement_yield1 = """                try:
                    yield (b'--frame\\r\\n'
                           b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')
                except GeneratorExit:
                    if cap is not None: cap.release()
                    raise"""

target_yield2 = """            yield (b'--frame\\r\\n'
                   b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')"""
replacement_yield2 = """            try:
                yield (b'--frame\\r\\n'
                       b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')
            except GeneratorExit:
                if cap is not None: cap.release()
                raise"""

# Replace all occurrences of these yield blocks 
# (there are 2 of the second type and 1 of the first type)
content = content.replace(target_yield1, replacement_yield1)
content = content.replace(target_yield2, replacement_yield2)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("App2.py successfully patched!")
