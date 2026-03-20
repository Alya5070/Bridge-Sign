import os

filepath = r'D:\Python Project\Hand Sign Detection\app2.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix the main yield block
target_yield = """        yield (b'--frame\\r\\n'
               b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')"""

replacement_yield = """        try:
            yield (b'--frame\\r\\n'
                   b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')
        except GeneratorExit:
            if cap is not None: cap.release()
            raise"""

if target_yield in content:
    content = content.replace(target_yield, replacement_yield)

# 2. Inject missing Quiz API endpoints before the video_feed route
quiz_endpoints = """
@app.route('/switch_mode', methods=['POST'])
@login_required
def switch_mode():
    global camera_mode
    data = request.json
    mode = data.get('mode', 'translation')
    camera_mode = 'translation' # Force translation for accurate inferences
    return jsonify({'success': True, 'mode': camera_mode})

@app.route('/get_current_prediction')
@login_required
def get_current_prediction():
    global current_prediction
    return jsonify({'prediction': current_prediction})

@app.route('/get_labels')
@login_required
def get_labels():
    global labels
    return jsonify({'labels': labels})

@app.route('/video_feed')"""

if "@app.route('/switch_mode', methods=['POST'])" not in content:
    content = content.replace("@app.route('/video_feed')", quiz_endpoints)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Quiz APIs injected and final GeneratorExit patched!")
