import pandas as pd
import googlemaps
import os

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_API_KEY', 'AIzaSyDUubz1oIkRHXOZqf-B_-6rd8DD18docvA'))

# Budget → price_level mapping
price_level_map = {500:0, 1000:1, 2000:2, 3000:3, 5000:4}

# Maximum selectable interests
MAX_INTERESTS = 3

# Function to fetch venues for multiple interests
def infer_and_fetch_multi():
    # Collect up to 3 interest types
    inp = input(f"Enter up to {MAX_INTERESTS} interest types (comma-separated): ")
    interests = [i.strip() for i in inp.split(',') if i.strip()][:MAX_INTERESTS]
    user_gender = input("Your gender (m/f/o): ")
    partner_gender = input("Partner gender (m/f/o): ")
    user_age = int(input("Your age: "))
    partner_age = int(input("Partner age: "))
    budget = int(input("Budget (500,1000,2000,3000,5000): "))
    district = input("District: ")

    price_level = price_level_map.get(budget, 0)
    seen = set()
    results = []

    # Query each interest and collect unique venues
    for interest in interests:
        query = f"{interest} in {district}, Bangkok"
        resp = gmaps.places(query=query, min_price=price_level, max_price=price_level, radius=5000)
        for place in resp.get('results', [])[:5]:
            pid = place.get('place_id')
            if pid in seen:
                continue
            seen.add(pid)
            results.append({
                'interest': interest,
                'name': place.get('name'),
                'address': place.get('formatted_address'),
                'rating': place.get('rating'),
                'link': f"https://www.google.com/maps/place/?q=place_id:{pid}"
            })

    # Display combined results
    print(f"\nRecommendations based on {interests}:\n")
    for idx, r in enumerate(results, 1):
        print(f"{idx}. [{r['interest']}] {r['name']} — {r['address']} (Rating: {r['rating']})")
        print(f"   Map: {r['link']}\n")

if __name__ == '__main__':
    infer_and_fetch_multi()