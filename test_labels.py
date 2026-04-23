import app2
app2.load_custom_model()
for lbl in app2.labels:
    print(repr(lbl), len(lbl))
