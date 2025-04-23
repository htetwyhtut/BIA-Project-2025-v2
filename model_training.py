import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from datetime import datetime
import re
import os
import random
from typing import List, Dict, Tuple, Any

class DateRecommendationSystem:
    def __init__(self):
        # Model file paths
        self.model_dir = "model_files"
        self.dataset_path = "training_dataset_v2.csv"
        self.encoder_path = os.path.join(self.model_dir, "encoders.joblib")
        self.scaler_path = os.path.join(self.model_dir, "scalers.joblib")
        self.interest_vectors_path = os.path.join(self.model_dir, "interest_vectors.joblib")
        self.place_df_path = os.path.join(self.model_dir, "processed_places.csv")
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Initialize components
        self.encoders = {}
        self.scalers = {}
        self.interest_vectors = {}
        self.place_df = None
        self.relationship_interest_mapping = self._create_relationship_interest_mapping()
        self.age_group_ranges = self._create_age_groups()
        
    def _create_relationship_interest_mapping(self) -> Dict[str, Dict[str, float]]:
        """
        Create weightings for different interests based on relationship type.
        Higher weights indicate stronger preference for that interest type given the relationship.
        """
        return {
            "romance": {
                "rooftop bar": 0.9, "fine dining": 0.85, "cocktail bar": 0.8, 
                "hidden bar": 0.75, "sky bar": 0.85, "dessert cafe": 0.7,
                "live music": 0.65, "movie theater": 0.6, "jazz bar": 0.7,
                "party": 0.5, "speakeasy": 0.75, "massage spa": 0.6,
                # Default weight for other interests
                "default": 0.3
            },
            "friends": {
                "party": 0.9, "clubbing": 0.85, "karaoke": 0.8, 
                "craft beer bar": 0.75, "arcade": 0.7, "board game cafe": 0.85,
                "live music": 0.8, "hot pot restaurant": 0.7, "bowling alley": 0.65,
                "escape room": 0.75, "hookah lounge": 0.6,
                "default": 0.4
            },
            "family": {
                "fine dining": 0.7, "dessert cafe": 0.9, "hot pot restaurant": 0.85,
                "ice cream shop": 0.8, "movie theater": 0.75, "live performance venue": 0.7,
                "theater": 0.65, "board game cafe": 0.6,
                "default": 0.3
            },
            "siblings": {
                "karaoke": 0.85, "arcade": 0.9, "movie theater": 0.8,
                "bowling alley": 0.85, "ice cream shop": 0.7, "escape room": 0.8,
                "board game cafe": 0.75, "hot pot restaurant": 0.7,
                "default": 0.4
            },
            "colleagues": {
                "craft beer bar": 0.8, "cocktail bar": 0.75, "rooftop bar": 0.7,
                "karaoke": 0.65, "hot pot restaurant": 0.8, "fine dining": 0.6,
                "default": 0.3
            },
            "default": {"default": 0.5}  # Default relationship type
        }

    def _create_age_groups(self) -> Dict[str, Tuple[int, int]]:
        """Define age group ranges"""
        return {
            "18-24": (18, 24),
            "25-34": (25, 34),
            "35-44": (35, 44),
            "45-54": (45, 54),
            "55-64": (55, 64),
            "65+": (65, 100)
        }
    
    def get_age_group(self, age: int) -> str:
        """Map age to age group"""
        for group, (min_age, max_age) in self.age_group_ranges.items():
            if min_age <= age <= max_age:
                return group
        return "25-34"  # Default
    
    def _extract_key_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and process key features from the dataset"""
        # Process types field to extract more specific categories
        df['types_list'] = df['types'].apply(lambda x: x.split(';') if isinstance(x, str) else [])
        
        # Create category flags based on types
        category_flags = [
            'casual_dining', 'fancy_dining', 'bar', 'nightlife',
            'activity', 'culture', 'outdoors', 'shopping'
        ]
        
        # Initialize columns
        for flag in category_flags:
            df[flag] = 0
            
        # Define mappings from Google place types to our categories
        type_to_category = {
            'restaurant': 'casual_dining',
            'food': 'casual_dining',
            'meal_takeaway': 'casual_dining',
            'cafe': 'casual_dining',
            'bar': 'bar',
            'night_club': 'nightlife',
            'movie_theater': 'activity',
            'museum': 'culture',
            'park': 'outdoors',
            'shopping_mall': 'shopping',
            'art_gallery': 'culture',
            'bakery': 'casual_dining',
            'spa': 'activity',
            'gym': 'activity',
            'amusement_park': 'activity'
        }
        
        # Apply the mappings
        for idx, row in df.iterrows():
            for place_type in row['types_list']:
                if place_type in type_to_category:
                    category = type_to_category[place_type]
                    df.at[idx, category] = 1
            
            # If restaurant has high price_level (3-4), mark as fancy_dining
            if ('restaurant' in row['types_list'] or 'meal_restaurant' in row['types_list']) and \
               row['price_level'] >= 3:
                df.at[idx, 'fancy_dining'] = 1
        
        return df
    
    def _calculate_sentiment_score(self, reviews: str) -> float:
        """Calculate a simple sentiment score from reviews"""
        if not isinstance(reviews, str) or not reviews:
            return 0.5  # Neutral default
            
        # Simple keyword-based sentiment analysis
        positive_keywords = ['great', 'excellent', 'amazing', 'good', 'best', 'love', 'wonderful',
                           'fantastic', 'perfect', 'outstanding', 'recommend', 'delicious']
        negative_keywords = ['bad', 'terrible', 'awful', 'worst', 'poor', 'disappoint', 'overpriced',
                           'avoid', 'mediocre', 'horrible', 'waste', 'rude']
                           
        # Convert to lowercase for case-insensitive matching
        reviews_lower = reviews.lower()
        
        # Count occurrences
        positive_count = sum(reviews_lower.count(word) for word in positive_keywords)
        negative_count = sum(reviews_lower.count(word) for word in negative_keywords)
        
        # Avoid division by zero
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.5
            
        # Calculate score (0 to 1)
        sentiment_score = positive_count / total_count
        
        return sentiment_score
    
    def preprocess_data(self, train: bool = True) -> pd.DataFrame:
        """Preprocess the dataset and create necessary features"""
        print("Loading and preprocessing data...")
        
        # Load dataset
        df = pd.read_csv(self.dataset_path)
        
        # Basic cleaning
        df = df.dropna(subset=['name', 'district', 'interest', 'latitude', 'longitude'])
        df['price_level'] = df['price_level'].fillna(1)  # Default price level
        df['rating'] = df['rating'].fillna(3.0)  # Default rating
        df['user_ratings_total'] = df['user_ratings_total'].fillna(10)  # Default number of ratings
        
        # Handle review text
        df['reviews'] = df['reviews'].fillna('')
        
        # Extract key features
        df = self._extract_key_features(df)
        
        # Calculate sentiment score from reviews
        df['sentiment_score'] = df['reviews'].apply(self._calculate_sentiment_score)
        
        # Normalize cost to a 0-1 scale if training
        if train:
            self.scalers['cost'] = MinMaxScaler()
            df['cost_normalized'] = self.scalers['cost'].fit_transform(df[['max_cost']])
        else:
            df['cost_normalized'] = self.scalers['cost'].transform(df[['max_cost']])
        
        # Create place feature vector (combining multiple aspects)
        feature_cols = [
            'rating', 'price_level', 'cost_normalized', 'sentiment_score',
            'casual_dining', 'fancy_dining', 'bar', 'nightlife',
            'activity', 'culture', 'outdoors', 'shopping'
        ]
        
        # One-hot encode district and interest if training
        if train:
            self.encoders['district'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            district_encoded = self.encoders['district'].fit_transform(df[['district']])
            district_cols = [f'district_{d}' for d in self.encoders['district'].categories_[0]]
            
            self.encoders['interest'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            interest_encoded = self.encoders['interest'].fit_transform(df[['interest']])
            interest_cols = [f'interest_{i}' for i in self.encoders['interest'].categories_[0]]
        else:
            district_encoded = self.encoders['district'].transform(df[['district']])
            district_cols = [f'district_{d}' for d in self.encoders['district'].categories_[0]]
            
            interest_encoded = self.encoders['interest'].transform(df[['interest']])
            interest_cols = [f'interest_{i}' for i in self.encoders['interest'].categories_[0]]
        
        # Add encoded columns to dataframe
        for i, col in enumerate(district_cols):
            df[col] = district_encoded[:, i]
            
        for i, col in enumerate(interest_cols):
            df[col] = interest_encoded[:, i]
        
        # Create a Google Maps URL for each place
        df['maps_url'] = df.apply(
            lambda row: f"https://www.google.com/maps/place/?q=place_id:{row['place_id']}", 
            axis=1
        )
        
        # Save processed data if training
        if train:
            self.place_df = df
            self.place_df.to_csv(self.place_df_path, index=False)
            joblib.dump(self.encoders, self.encoder_path)
            joblib.dump(self.scalers, self.scaler_path)
        
        return df
        
    def build_interest_vectors(self) -> Dict[str, np.ndarray]:
        """Build interest-based feature vectors for content-based filtering"""
        print("Building interest vectors...")
        
        # Get all unique interests
        interests = self.place_df['interest'].unique()
        
        # For each interest, create a feature vector
        interest_vectors = {}
        
        for interest in interests:
            # Get places with this interest
            interest_places = self.place_df[self.place_df['interest'] == interest]
            
            if len(interest_places) > 0:
                # Calculate average feature values for this interest
                feature_cols = [
                    'rating', 'price_level', 'sentiment_score',
                    'casual_dining', 'fancy_dining', 'bar', 'nightlife',
                    'activity', 'culture', 'outdoors', 'shopping'
                ]
                
                interest_vector = interest_places[feature_cols].mean().values
                interest_vectors[interest] = interest_vector
        
        # Save the interest vectors
        self.interest_vectors = interest_vectors
        joblib.dump(interest_vectors, self.interest_vectors_path)
        
        return interest_vectors
        
    def train(self) -> None:
        """Train the recommendation model"""
        print("Training recommendation model...")
        
        # Process the dataset
        self.preprocess_data(train=True)
        
        # Build interest vectors for content-based filtering
        self.build_interest_vectors()
        
        print("Model training completed.")
    
    def load_model(self) -> None:
        """Load the trained model components"""
        print("Loading model components...")
        
        # Load encoders and scalers
        if os.path.exists(self.encoder_path) and os.path.exists(self.scaler_path):
            self.encoders = joblib.load(self.encoder_path)
            self.scalers = joblib.load(self.scaler_path)
        else:
            raise FileNotFoundError("Model files not found. Please train the model first.")
        
        # Load interest vectors for content-based filtering
        if os.path.exists(self.interest_vectors_path):
            self.interest_vectors = joblib.load(self.interest_vectors_path)
        else:
            raise FileNotFoundError("Interest vectors not found. Please train the model first.")
        
        # Load processed place data
        if os.path.exists(self.place_df_path):
            self.place_df = pd.read_csv(self.place_df_path)
        else:
            raise FileNotFoundError("Processed place data not found. Please train the model first.")
        
        print("Model components loaded successfully.")
    
    SCORING_WEIGHTS = {
    'similarity': 0.4,
    'interest_match': 0.3,
    'rating': 0.2,
    'budget_alignment': 0.05,
    'randomness': 0.05}  # Optional: Set to 0 to disable
    
    def get_recommendations(self, 
                          relationship_type: str, 
                          user_gender: str, 
                          partner_gender: str,
                          user_age: int, 
                          partner_age: int, 
                          interests: List[str],
                          budget: int,
                          district: str = None,
                          num_recommendations: int = 8) -> pd.DataFrame:
        """
        Generate place recommendations based on user inputs
        
        Args:
            relationship_type: Type of relationship (romance, friends, family, etc.)
            user_gender: User's gender (m, f, o)
            partner_gender: Partner's gender (m, f, o)
            user_age: User's age (18-100)
            partner_age: Partner's age (18-100)
            interests: List of interests/place types
            budget: Maximum budget (in THB)
            district: Preferred district (optional)
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended places
        """
        # Make sure model is loaded
        if self.place_df is None:
            try:
                self.load_model()
            except FileNotFoundError:
                print("Model not trained. Training now...")
                self.train()
        
        # Process input parameters
        user_age_group = self.get_age_group(user_age)
        partner_age_group = self.get_age_group(partner_age)
        
        # Get relationship type weights (or use default)
        if relationship_type.lower() in self.relationship_interest_mapping:
            rel_weights = self.relationship_interest_mapping[relationship_type.lower()]
        else:
            rel_weights = self.relationship_interest_mapping["default"]
        
        # Filter based on budget
        budget_normalized = self.scalers['cost'].transform([[budget]])[0][0]
        budget_filtered_df = self.place_df[self.place_df['max_cost'] <= budget]
        
        # If no places within budget, take the 25% cheapest places
        if len(budget_filtered_df) < 10:
            budget_threshold = self.place_df['max_cost'].quantile(0.25)
            budget_filtered_df = self.place_df[self.place_df['max_cost'] <= budget_threshold]
        
        # Filter by district if specified
        if district and f"district_{district}" in budget_filtered_df.columns:
            district_filtered_df = budget_filtered_df[budget_filtered_df[f"district_{district}"] == 1]
            # If too few places in specified district, use all districts
            if len(district_filtered_df) < 5:
                district_filtered_df = budget_filtered_df
        else:
            district_filtered_df = budget_filtered_df
            
        # Apply content-based filtering using interest vectors
        # Calculate scores for each place based on interest similarity and relationship weights
        interest_scores = []
        
        # Create a base vector representing interests provided by user
        user_interest_vector = np.zeros(11)  # Same length as our interest vectors
        
        # Get available interest vectors based on specified interests
        available_interests = [i for i in interests if i in self.interest_vectors]
        
        # If no valid interests specified, use all interests with rel_weights
        if not available_interests:
            available_interests = list(self.interest_vectors.keys())
        
        # Apply interest weights
        for interest in available_interests:
            # Get weight (either specific to relationship type or default)
            weight = rel_weights.get(interest, rel_weights.get("default", 0.5))
            
            # If interest vector exists, add it to user vector with appropriate weight
            if interest in self.interest_vectors:
                user_interest_vector += weight * self.interest_vectors[interest]
        
        # Normalize the vector
        if np.sum(user_interest_vector) > 0:
            user_interest_vector = user_interest_vector / np.sum(user_interest_vector)
        
        # Calculate similarity scores
        for idx, row in district_filtered_df.iterrows():
            # Build place vector
            place_features = [
                row['rating'], row['price_level'], row['sentiment_score'],
                row['casual_dining'], row['fancy_dining'], row['bar'], row['nightlife'],
                row['activity'], row['culture'], row['outdoors'], row['shopping']
            ]
            place_vector = np.array(place_features)
            
            # Calculate direct interest match boost
            interest_match_boost = 0
            for interest in interests:
                interest_col = f"interest_{interest}"
                if interest_col in row.index and row[interest_col] == 1:
                    # Apply relationship weight if available
                    interest_match_boost += rel_weights.get(interest, rel_weights.get("default", 0.5))
            
            # Calculate cosine similarity
            if np.sum(place_vector) > 0:  # Avoid division by zero
                similarity = cosine_similarity([user_interest_vector], [place_vector])[0][0]
            else:
                similarity = 0
                
            # Combined score with various factors
            # - Base similarity from interests
            # - Bonus for direct interest matches
            # - Rating factor
            # - Small random component for variety
            combined_score = (
                0.4 * similarity + 
                0.3 * interest_match_boost + 
                0.2 * (row['rating'] / 5.0) + 
                0.05 * (1 - abs(budget_normalized - row['cost_normalized'])) +
                0.05 * random.random()  # Add slight randomness
            )
            
            
            interest_scores.append((idx, combined_score))
        
        # Sort by score and get top recommendations
        sorted_scores = sorted(interest_scores, key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sorted_scores[:num_recommendations*2]]  # Get more for diversity
        
        # Get recommended places
        recommendations = district_filtered_df.loc[top_indices].copy()
        
        # Apply diversity filter
        # Ensure variety in types of places
        diverse_recommendations = self._ensure_diversity(recommendations, num_recommendations)
        
        # Select final columns
        result_columns = ['name', 'district', 'interest', 'rating', 'max_cost', 'maps_url']
        final_recommendations = diverse_recommendations[result_columns].head(num_recommendations)
        
        return final_recommendations
    
    def _ensure_diversity(self, recommendations: pd.DataFrame, num_recommendations: int) -> pd.DataFrame:
        """Ensure diversity in recommendations by avoiding too many of same type"""
        if len(recommendations) <= num_recommendations:
            return recommendations
            
        # Prioritize by interest diversity
        selected_indices = []
        selected_interests = set()
        
        # First pass: try to get diverse interests
        for idx in recommendations.index:
            interest = recommendations.loc[idx, 'interest']
            if interest not in selected_interests and len(selected_indices) < num_recommendations:
                selected_indices.append(idx)
                selected_interests.add(interest)
                
        # Second pass: fill remaining slots
        remaining_slots = num_recommendations - len(selected_indices)
        if remaining_slots > 0:
            unselected = [idx for idx in recommendations.index if idx not in selected_indices]
            selected_indices.extend(unselected[:remaining_slots])
        
        return recommendations.loc[selected_indices]
    
    def evaluate_model(self, test_scenarios: List[Dict[str, Any]] = None) -> None:
        """
        Evaluate the model using test scenarios or default scenarios
        
        Args:
            test_scenarios: List of test user scenarios
        """
        if test_scenarios is None:
            # Default test scenarios
            test_scenarios = [
                {
                    "scenario": "Romantic date for young couple",
                    "params": {
                        "relationship_type": "romance",
                        "user_gender": "m",
                        "partner_gender": "f",
                        "user_age": 28,
                        "partner_age": 26,
                        "interests": ["rooftop bar", "fine dining", "cocktail bar"],
                        "budget": 3000,
                        "district": None
                    }
                },
                {
                    "scenario": "Friend hangout with limited budget",
                    "params": {
                        "relationship_type": "friends",
                        "user_gender": "f",
                        "partner_gender": "f",
                        "user_age": 22,
                        "partner_age": 23,
                        "interests": ["cafe", "board game cafe", "karaoke"],
                        "budget": 1000,
                        "district": None
                    }
                },
                {
                    "scenario": "Family outing with children",
                    "params": {
                        "relationship_type": "family",
                        "user_gender": "m",
                        "partner_gender": "f",
                        "user_age": 40,
                        "partner_age": 38,
                        "interests": ["dessert cafe", "ice cream shop", "movie theater"],
                        "budget": 2000,
                        "district": None
                    }
                }
            ]
        
        print("Evaluating model with test scenarios...")
        
        for scenario in test_scenarios:
            print(f"\nScenario: {scenario['scenario']}")
            recommendations = self.get_recommendations(**scenario['params'])
            
            print(f"Found {len(recommendations)} recommendations:")
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. {row['name']} - {row['interest']} in {row['district']}")
                print(f"   Rating: {row['rating']:.1f}, Cost: {row['max_cost']} THB")
                print(f"   Maps: {row['maps_url']}")
                print()
                
            # Check diversity of results
            interest_diversity = len(recommendations['interest'].unique()) / len(recommendations)
            district_diversity = len(recommendations['district'].unique()) / len(recommendations)
            
            print(f"Interest diversity: {interest_diversity:.2f}")
            print(f"District diversity: {district_diversity:.2f}")
            print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Create and train the recommendation system
    recommender = DateRecommendationSystem()
    
    # Training
    try:
        recommender.load_model()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model not found. Training new model...")
        recommender.train()
    
    # Evaluate
    recommender.evaluate_model()
    
    # Example inference
    print("\nExample recommendation:")
    recommendations = recommender.get_recommendations(
        relationship_type="romance",
        user_gender="m",
        partner_gender="f",
        user_age=30,
        partner_age=28,
        interests=["fine dining", "cocktail bar"],
        budget=3000,
        district=None
    )
    
    print(f"Top recommendations:")
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {row['name']} ({row['interest']})")
        print(f"   Rating: {row['rating']:.1f}, Max Cost: {row['max_cost']} THB")
        print(f"   Location: {row['district']}")
        print(f"   Google Maps: {row['maps_url']}")
        print()