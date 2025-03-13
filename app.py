from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

app = Flask(__name__)

class PokemonCardRecommender:
    def __init__(self, decks_path, cards_path):
        """Initialize the recommender system with deck and card data."""
        self.decks_df = pd.read_csv(decks_path)
        self.cards_df = pd.read_csv(cards_path)

        # Create card-deck matrix (rows: decks, columns: cards, values: card count)
        self.card_deck_matrix = self.decks_df.pivot_table(
            index="id", columns="card_id", values="card_count", fill_value=0
        )

        # Create mapping from card_id to card details
        self.card_id_to_details = {}
        for _, card in self.cards_df.iterrows():
            self.card_id_to_details[card["id"]] = {
                "name": card["name"],
                "types": card["types"],
                "supertype": card["supertype"],
                "subtypes": card["subtypes"],
                "evolvesTo": card["evolvesTo"]
            }

        # Pre-compute card similarity matrix
        print("Computing card similarity matrix...")
        self.card_similarity_df = self._compute_similarity_matrix()
        print("Similarity matrix computed.")

        # Define card synergies and type effectiveness
        self.type_effectiveness = self._define_type_effectiveness()

    def _compute_similarity_matrix(self):
        """Compute the card similarity matrix using cosine similarity."""
        card_similarity_matrix = cosine_similarity(self.card_deck_matrix.T)
        return pd.DataFrame(
            card_similarity_matrix,
            index=self.card_deck_matrix.columns,
            columns=self.card_deck_matrix.columns
        )

    def _define_type_effectiveness(self):
        """Define basic type effectiveness for Pokémon TCG."""
        return {
            "Fire": {
                "strong_against": ["Grass", "Bug", "Steel"],
                "weak_against": ["Water", "Rock"]
            },
            "Water": {
                "strong_against": ["Fire", "Ground"],
                "weak_against": ["Electric", "Grass"]
            },
            "Grass": {
                "strong_against": ["Water", "Ground"],
                "weak_against": ["Fire", "Flying", "Bug"]
            },
            "Electric": {
                "strong_against": ["Water", "Flying"],
                "weak_against": ["Ground"]
            },
            "Fighting": {
                "strong_against": ["Normal", "Rock", "Steel", "Dark"],
                "weak_against": ["Flying", "Psychic"]
            },
            "Psychic": {
                "strong_against": ["Fighting", "Poison"],
                "weak_against": ["Dark", "Bug"]
            }
        }

    def _parse_types(self, types_str):
        """Parse the types string from the card details."""
        if not types_str or pd.isna(types_str):
            return []

        try:
            if isinstance(types_str, str):
                if types_str.startswith('[') and types_str.endswith(']'):
                    return eval(types_str)
                else:
                    return [t.strip() for t in types_str.split(',')]
            elif isinstance(types_str, list):
                return types_str
        except:
            pass

        return []

    def _parse_subtypes(self, subtypes_str):
        """Parse the subtypes string from the card details."""
        return self._parse_types(subtypes_str)

    def _analyze_deck(self, input_cards):
        """Analyze the deck to identify main types and composition."""
        pokemon_types = Counter()
        card_supertypes = Counter()

        for card_id in input_cards:
            if card_id in self.card_id_to_details:
                details = self.card_id_to_details[card_id]
                supertype = details.get("supertype", "")
                card_supertypes[supertype] += 1

                if supertype == "Pokémon":
                    types = self._parse_types(details.get("types", []))
                    for t in types:
                        pokemon_types[t] += 1

        main_types = []
        if pokemon_types:
            sorted_types = sorted(pokemon_types.items(), key=lambda x: x[1], reverse=True)
            if sorted_types:
                main_types.append(sorted_types[0][0])  # Primary type
                if len(sorted_types) > 1 and sorted_types[1][1] >= 2:
                    main_types.append(sorted_types[1][0])  # Secondary type

        total_cards = len(input_cards)
        deck_balance = {
            "pokemon": card_supertypes.get("Pokémon", 0) / total_cards if total_cards else 0,
            "trainer": card_supertypes.get("Trainer", 0) / total_cards if total_cards else 0,
            "energy": card_supertypes.get("Energy", 0) / total_cards if total_cards else 0
        }

        return {
            "main_types": main_types,
            "type_counts": pokemon_types,
            "deck_balance": deck_balance
        }

    def _get_content_based_scores(self, input_cards):
        """Get content-based similarity scores for cards based on deck co-occurrence."""
        valid_cards = [card for card in input_cards if card in self.card_similarity_df.columns]
        if not valid_cards:
            return pd.Series()

        card_scores = self.card_similarity_df[valid_cards].sum(axis=1)
        return card_scores.drop(input_cards, errors="ignore")

    def _get_type_based_scores(self, input_cards, deck_analysis, weight=1.0):
        """Generate scores based on type matching and effectiveness."""
        scores = {}
        main_types = deck_analysis["main_types"]
        if not main_types:
            return scores

        for card_id, details in self.card_id_to_details.items():
            if card_id not in input_cards:
                supertype = details.get("supertype", "")
                if supertype == "Pokémon":
                    card_types = self._parse_types(details.get("types", []))
                    score = 0
                    for card_type in card_types:
                        if card_type in main_types:
                            score += 2.0
                        for main_type in main_types:
                            if main_type in self.type_effectiveness:
                                weak_against = self.type_effectiveness[main_type]["weak_against"]
                                if card_type in self.type_effectiveness:
                                    strong_against = self.type_effectiveness[card_type]["strong_against"]
                                    if any(weak_type in strong_against for weak_type in weak_against):
                                        score += 1.0
                    if score > 0:
                        scores[card_id] = score * weight
                elif supertype == "Energy":
                    name = details.get("name", "")
                    for main_type in main_types:
                        if main_type in name:
                            scores[card_id] = 2.0 * weight
                            break
        return scores

    def _get_balance_scores(self, input_cards, deck_analysis, weight=1.0):
        """Generate scores to balance the deck composition."""
        scores = {}
        balance = deck_analysis["deck_balance"]
        ideal = {"pokemon": 0.4, "trainer": 0.4, "energy": 0.2}
        need_more_pokemon = balance["pokemon"] < ideal["pokemon"] - 0.05
        need_more_trainers = balance["trainer"] < ideal["trainer"] - 0.05
        need_more_energy = balance["energy"] < ideal["energy"] - 0.05

        for card_id, details in self.card_id_to_details.items():
            if card_id not in input_cards:
                supertype = details.get("supertype", "")
                if supertype == "Pokémon" and need_more_pokemon:
                    card_types = self._parse_types(details.get("types", []))
                    if any(t in deck_analysis["main_types"] for t in card_types):
                        scores[card_id] = 1.5 * weight
                elif supertype == "Trainer" and need_more_trainers:
                    name = details.get("name", "")
                    if any(key in name for key in ["Professor", "Ball", "Search", "Switch", "Potion"]):
                        scores[card_id] = 1.8 * weight
                    else:
                        scores[card_id] = 1.2 * weight
                elif supertype == "Energy" and need_more_energy:
                    name = details.get("name", "")
                    if any(t in name for t in deck_analysis["main_types"]):
                        scores[card_id] = 2.0 * weight
        return scores

    def _get_synergy_scores(self, input_cards, weight=1.0):
        """Generate scores based on card synergies."""
        synergy_groups = {
            "draw_support": ["Professor", "N", "Cheren", "Bianca", "Colress"],
            "search": ["Ball", "Search", "Communication"],
            "recovery": ["Potion", "Switch", "Revive"],
            "energy_support": ["Energy Retrieval", "Energy Search"]
        }
        synergy_counts = {group: 0 for group in synergy_groups}
        for card_id in input_cards:
            if card_id in self.card_id_to_details:
                name = self.card_id_to_details[card_id].get("name", "")
                for group, keywords in synergy_groups.items():
                    if any(kw in name for kw in keywords):
                        synergy_counts[group] += 1
        scores = {}
        for card_id, details in self.card_id_to_details.items():
            if card_id not in input_cards:
                name = details.get("name", "")
                supertype = details.get("supertype", "")
                if supertype == "Trainer":
                    for group, keywords in synergy_groups.items():
                        if any(kw in name for kw in keywords):
                            if synergy_counts[group] < 2:
                                scores[card_id] = 2.0 * weight
                            else:
                                scores[card_id] = 1.0 * weight
        return scores

    def _generate_explanation(self, card_id, deck_analysis):
        """Generate an explanation for why this card is recommended."""
        details = self.card_id_to_details.get(card_id, {})
        supertype = details.get("supertype", "")
        name = details.get("name", "")
        if supertype == "Pokémon":
            types = self._parse_types(details.get("types", []))
            if any(t in deck_analysis["main_types"] for t in types):
                return f"Strengthens your {'/'.join(deck_analysis['main_types'])} strategy"
            else:
                return "Provides type coverage for your deck"
        elif supertype == "Trainer":
            if any(draw in name for draw in ["Professor", "N", "Cheren", "Bianca"]):
                return "Adds essential draw power to your deck"
            elif any(search in name for search in ["Ball", "Search", "Communication"]):
                return "Improves consistency with search capability"
            elif "Potion" in name or "Revive" in name:
                return "Provides recovery options for your Pokémon"
            elif "Energy" in name:
                return "Helps manage your energy cards"
            else:
                return "Adds strategic utility to your deck"
        elif supertype == "Energy":
            if any(t in name for t in deck_analysis["main_types"]):
                return f"Provides energy for your {'/'.join(deck_analysis['main_types'])} Pokémon"
            else:
                return "Adds energy support for your Pokémon"
        return "Complements your current deck strategy"

    def recommend_cards(self, current_deck, top_n=10):
        """Recommend cards based on the current deck using a hybrid approach."""
        if not current_deck:
            return []

        deck_analysis = self._analyze_deck(current_deck)
        content_scores = self._get_content_based_scores(current_deck)
        type_scores = pd.Series(self._get_type_based_scores(current_deck, deck_analysis, weight=1.2))
        balance_scores = pd.Series(self._get_balance_scores(current_deck, deck_analysis, weight=0.8))
        synergy_scores = pd.Series(self._get_synergy_scores(current_deck, weight=1.0))

        final_scores = pd.Series(0, index=self.card_similarity_df.index)
        for scores in [content_scores, type_scores, balance_scores, synergy_scores]:
            if not scores.empty:
                final_scores = final_scores.add(scores, fill_value=0)

        recommended_card_ids = final_scores.nlargest(top_n).index.tolist()
        recommendations = []
        for i, card_id in enumerate(recommended_card_ids):
            card_details = self.card_id_to_details.get(card_id, {})
            explanation = self._generate_explanation(card_id, deck_analysis)
            recommendations.append({
                "rank": i + 1,
                "card_id": card_id,
                "card_name": card_details.get("name", "Unknown"),
                "types": card_details.get("types", ""),
                "score": float(final_scores.get(card_id, 0)),
                "explanation": explanation
            })
        return recommendations

# Initialize the recommender system
recommender = PokemonCardRecommender("decks.csv", "cards.csv")

# Flask endpoints
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    # Expecting a comma-separated list of card IDs (e.g., "bw1-19, bw1-17, bw1-15")
    deck_str = data.get('deck', '')
    if not deck_str:
        return jsonify({'error': 'No deck input provided.'}), 400

    current_deck = [card.strip() for card in deck_str.split(',') if card.strip()]
    recommendations = recommender.recommend_cards(current_deck, top_n=10)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
