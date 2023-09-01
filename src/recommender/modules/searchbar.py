import yaml
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('../../data/title.yml', 'r') as file:
    titles = yaml.load(file, Loader=yaml.FullLoader)

inverted_titles = {v: k for k, v in titles.items()}

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip().lower()
    suggestions = [title for title in titles.values() if query in title.lower()]

    return jsonify(suggestions)

@app.route('/get_label', methods=['GET'])
def get_label():
    selected_title = request.args.get('title', '').strip()
    label = inverted_titles.get(selected_title, "Not found")

    return jsonify({"label": label})

if __name__ == '__main__':
    app.run(debug=True)

