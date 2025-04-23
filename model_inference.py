import joblib
import pandas as pd
import googlemaps
import os

# 1) Load trained model and encoder
pipeline = joblib.load('model_pipeline.joblib')
le       = joblib.load('label_encoder.joblib')

# 2) Initialize Google Maps client (replace with your API key or env var)
API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDUubz1oIkRHXOZqf-B_-6rd8DD18docvA')
gmaps = googlemaps.Client(key=API_KEY)

# 3) Helper to bin ages
def bin_age(age):
    bins = [18,25,35,45,60,100]
    labels = ['18-24','25-34','35-44','45-59','60+']
    return pd.cut([age], bins=bins, labels=labels, right=False)[0]

# 4) Main recommendation function
def recommend_venues(user_input, top_n=5):
    """
    user_input: dict with keys:
      - relationship_type, user_gender, partner_gender,
      - user_age, partner_age, budget, district
    """
    # Prepare DataFrame
    df = pd.DataFrame([user_input])
    df['user_age_bin']    = df['user_age'].apply(bin_age)
    df['partner_age_bin'] = df['partner_age'].apply(bin_age)

    features = df[[
        'relationship_type','user_gender','partner_gender',
        'user_age_bin','partner_age_bin','budget','district'
    ]]

    # Predict interest category
    pred_idx = pipeline.predict(features)
    category = le.inverse_transform(pred_idx)[0]

    # Map budget to price level (0-4 scale for Google API)
    # Example mapping: 500->0, 1000->1, 2000->2, 3000->3, 5000->4
    price_map = {500:0, 1000:1, 2000:2, 3000:3, 5000:4}
    budget = user_input.get('budget', 500)
    min_price = price_map.get(budget, 0)
    max_price = price_map.get(budget, 4)

    # Query Google Places API
    query = f"{category} in {user_input['district']}, Bangkok"
    resp = gmaps.places(
        query=query,
        min_price=min_price,
        max_price=max_price,
        radius=5000
    )

    results = []
    for place in resp.get('results', [])[:top_n]:
        place_id = place.get('place_id')
        maps_link = (
            f"https://www.google.com/maps/search/?api=1"
            f"&query_place_id={place_id}"
        )
        results.append({
            'name':    place.get('name'),
            'address': place.get('formatted_address'),
            'rating':  place.get('rating'),
            'link':    maps_link
        })

    return {
        'predicted_interest': category,
        'venues': results
    }

# 5) CLI example
def main():
    # Collect user input via console (or replace with function args)
    ui = {
        'relationship_type': input("Relationship type: "),
        'user_gender':       input("Your gender (m/f/o): "),
        'partner_gender':    input("Partner gender (m/f/o): "),
        'user_age':          int(input("Your age: ")),
        'partner_age':       int(input("Partner age: ")),
        'budget':            int(input("Budget (e.g. 500,1000,2000,3000,5000): ")),
        'district':          input("District: ")
    }

    recs = recommend_venues(ui, top_n=5)
    print("Predicted interest category:", recs['predicted_interest'])
    print("Top venue recommendations:")
    for i, v in enumerate(recs['venues'], 1):
        print(f"{i}. {v['name']} — {v['address']} (Rating: {v['rating']})")
        print(f"   Map: {v['link']}")

if __name__ == '__main__':
    main()