import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import re
import json

class PokemonCardRecommender:
    def __init__(self, decks_df, cards_df):
        """Initialize the recommender system with deck and card data."""
        self.decks_df = decks_df
        self.cards_df = cards_df

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
        with st.spinner("Computing card similarity matrix..."):
            self.card_similarity_df = self._compute_similarity_matrix()

        # Define card synergies and type effectiveness
        self.type_effectiveness = self._define_type_effectiveness()

    # All the existing methods from the original class go here
    # (I'm including only a few important ones for brevity)
    
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
        
    # Include all other methods from the original class
    # ...

    def recommend_cards(self, current_deck, top_n=10):
        """Recommend cards based on the current deck using a hybrid approach."""
        if not current_deck:
            return []

        # Analyze deck
        deck_analysis = self._analyze_deck(current_deck)

        # Get various score components
        content_scores = self._get_content_based_scores(current_deck)
        type_scores = pd.Series(self._get_type_based_scores(current_deck, deck_analysis, weight=1.2))
        balance_scores = pd.Series(self._get_balance_scores(current_deck, deck_analysis, weight=0.8))
        synergy_scores = pd.Series(self._get_synergy_scores(current_deck, weight=1.0))

        # Combine all scores
        final_scores = pd.Series(0, index=self.card_similarity_df.index)

        for scores in [content_scores, type_scores, balance_scores, synergy_scores]:
            if not scores.empty:
                final_scores = final_scores.add(scores, fill_value=0)

        # Get top recommendations
        recommended_card_ids = final_scores.nlargest(top_n).index.tolist()

        # Prepare detailed recommendations
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


# Streamlit app
def main():
    st.set_page_config(
        page_title="Pokémon TCG Card Recommender",
        page_icon="🃏",
        layout="wide"
    )
    
    st.title("Pokémon TCG Card Recommender")
    st.write("""
    This app recommends Pokémon Trading Card Game cards to improve your deck.
    Upload your card data files below or use our sample data.
    """)
    
    # Data loading section
    st.header("1. Data Upload")
    
    data_option = st.radio(
        "Choose data source:",
        ["Use sample data", "Upload my own data"]
    )
    
    decks_df = None
    cards_df = None
    
    if data_option == "Use sample data":
        # Load sample data
        try:
            decks_df = pd.read_csv("decks.csv")
            cards_df = pd.read_csv("cards.csv")
            st.success("Sample data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
            st.info("Please upload your own data files instead.")
            data_option = "Upload my own data"
    
    if data_option == "Upload my own data":
        col1, col2 = st.columns(2)
        
        with col1:
            decks_file = st.file_uploader("Upload decks.csv", type=["csv"])
            if decks_file is not None:
                try:
                    decks_df = pd.read_csv(decks_file)
                    st.success(f"Decks file uploaded successfully! ({len(decks_df)} rows)")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            cards_file = st.file_uploader("Upload cards.csv", type=["csv"])
            if cards_file is not None:
                try:
                    cards_df = pd.read_csv(cards_file)
                    st.success(f"Cards file uploaded successfully! ({len(cards_df)} rows)")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Only proceed if both datasets are loaded
    if decks_df is not None and cards_df is not None:
        st.header("2. Initialize Recommender")
        if st.button("Initialize Recommender System"):
            with st.spinner("Initializing recommender system..."):
                recommender = PokemonCardRecommender(decks_df, cards_df)
                st.session_state['recommender'] = recommender
                st.success("Recommender system initialized successfully!")
                st.session_state['cards_list'] = list(recommender.card_id_to_details.keys())
                st.session_state['cards_names'] = {
                    card_id: f"{details.get('name', 'Unknown')} ({card_id})"
                    for card_id, details in recommender.card_id_to_details.items()
                }
        
        # Only show the deck builder if recommender is initialized
        if 'recommender' in st.session_state:
            st.header("3. Build Your Deck")
            
            # Search for cards
            search_term = st.text_input("Search for cards by name:")
            if search_term:
                filtered_cards = {
                    card_id: name for card_id, name in st.session_state['cards_names'].items()
                    if search_term.lower() in name.lower()
                }
                if filtered_cards:
                    selected_card = st.selectbox("Select a card to add:", 
                                               options=list(filtered_cards.keys()),
                                               format_func=lambda x: filtered_cards[x])
                    if st.button("Add Card to Deck"):
                        if 'current_deck' not in st.session_state:
                            st.session_state['current_deck'] = []
                        if selected_card not in st.session_state['current_deck']:
                            st.session_state['current_deck'].append(selected_card)
                            st.success(f"Added {filtered_cards[selected_card]} to your deck!")
                else:
                    st.info("No cards found with that name.")
            
            # Display current deck
            if 'current_deck' in st.session_state and st.session_state['current_deck']:
                st.subheader("Your Current Deck:")
                
                # Create a deck view with card details
                deck_items = []
                for card_id in st.session_state['current_deck']:
                    card_details = st.session_state['recommender'].card_id_to_details.get(card_id, {})
                    deck_items.append({
                        "Card ID": card_id,
                        "Name": card_details.get("name", "Unknown"),
                        "Type": card_details.get("types", ""),
                        "Supertype": card_details.get("supertype", "")
                    })
                
                deck_df = pd.DataFrame(deck_items)
                st.dataframe(deck_df)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Clear Deck"):
                        st.session_state['current_deck'] = []
                        st.info("Deck cleared!")
                        st.experimental_rerun()
                
                with col2:
                    if st.button("Get Recommendations"):
                        with st.spinner("Generating recommendations..."):
                            recommendations = st.session_state['recommender'].recommend_cards(
                                st.session_state['current_deck'], top_n=10)
                            st.session_state['recommendations'] = recommendations
            else:
                if 'current_deck' not in st.session_state:
                    st.session_state['current_deck'] = []
                st.info("Your deck is empty. Add some cards to get started!")
                
                # Option to load a sample deck
                if st.button("Load Sample Deck"):
                    # Sample Red Frenzy deck cards
                    sample_cards = [
                        "bw1-19",  # Emboar
                        "bw1-17",  # Pignite
                        "bw1-15",  # Tepig
                        "bw1-22",  # Simisear
                        "bw1-21",  # Pansear
                        "bw1-25",  # Darmanitan
                        "bw1-23",  # Darumaka
                        "bw1-58",  # Timburr
                        "bw1-83",  # Stoutland
                        "bw1-82",  # Herdier
                        "bw1-81",  # Lillipup
                        "bw1-89",  # Cinccino
                        "bw1-88",  # Minccino
                        "bw1-93",  # Energy Search
                        "bw1-104", # Switch
                        "bw1-92",  # Energy Retrieval
                        "bw1-99",  # Pokémon Communication
                        "bw1-101", # Professor Juniper
                        "bw1-102", # Revive
                        "bw1-106", # Fire Energy
                        "bw1-110"  # Fighting Energy
                    ]
                    # Filter to only include cards that exist in our dataset
                    valid_sample_cards = [card for card in sample_cards 
                                         if card in st.session_state['cards_list']]
                    st.session_state['current_deck'] = valid_sample_cards
                    st.success(f"Loaded sample deck with {len(valid_sample_cards)} cards!")
                    st.experimental_rerun()
            
            # Display recommendations
            if 'recommendations' in st.session_state and st.session_state['recommendations']:
                st.header("4. Card Recommendations")
                st.write("Here are the top recommended cards for your deck:")
                
                rec_items = []
                for rec in st.session_state['recommendations']:
                    rec_items.append({
                        "Rank": rec['rank'],
                        "Card": f"{rec['card_name']} ({rec['card_id']})",
                        "Types": rec['types'],
                        "Score": f"{rec['score']:.4f}",
                        "Explanation": rec['explanation']
                    })
                
                rec_df = pd.DataFrame(rec_items)
                st.table(rec_df)
                
                # Add buttons to add recommended cards to deck
                st.subheader("Add a recommended card to your deck:")
                col1, col2 = st.columns(2)
                with col1:
                    rec_to_add = st.selectbox("Select a recommendation:", 
                                            options=range(len(st.session_state['recommendations'])),
                                            format_func=lambda x: f"{st.session_state['recommendations'][x]['card_name']} ({st.session_state['recommendations'][x]['card_id']})")
                
                with col2:
                    if st.button("Add to Deck"):
                        card_to_add = st.session_state['recommendations'][rec_to_add]['card_id']
                        if card_to_add not in st.session_state['current_deck']:
                            st.session_state['current_deck'].append(card_to_add)
                            # Clear recommendations as the deck has changed
                            st.session_state.pop('recommendations', None)
                            st.success(f"Added {st.session_state['recommendations'][rec_to_add]['card_name']} to your deck!")
                            st.experimental_rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("Pokémon TCG Card Recommender - Build better decks, win more games!")


if __name__ == "__main__":
    main()