import os
from flask import Flask, request, jsonify, render_template
from model_training import DateRecommendationSystem

app = Flask(__name__, static_folder='static', template_folder='templates')
recommender = DateRecommendationSystem()

# Load or train model at startup
try:
    recommender.load_model()
except FileNotFoundError:
    recommender.train()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/interests')
def get_interests():
    if recommender.place_df is not None:
        return jsonify(interests=sorted(recommender.place_df['interest'].unique().tolist()))
    return jsonify(interests=[])

@app.route('/api/districts')
def get_districts():
    if recommender.place_df is not None:
        return jsonify(districts=sorted(recommender.place_df['district'].unique().tolist()))
    return jsonify(districts=[])

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    recs = recommender.get_recommendations(
        relationship_type = data.get('relationship_type', 'friends'),
        user_gender       = data.get('user_gender', 'm'),
        partner_gender    = data.get('partner_gender', 'f'),
        user_age          = int(data.get('user_age', 25)),
        partner_age       = int(data.get('partner_age', 25)),
        interests         = data.get('interests', []),
        budget            = int(data.get('budget', 2000)),
        district          = data.get('district')
    )
    out = []
    for _, r in recs.iterrows():
        out.append({
            'name':      r['name'],
            'district':  r['district'],
            'interest':  r['interest'],
            'rating':    float(r['rating']),
            'max_cost':  float(r['max_cost']),
            'maps_url':  r['maps_url']
        })
    return jsonify(recommendations=out)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
