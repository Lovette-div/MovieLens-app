# -*- coding: utf-8 -*-
"""
================================================================================
MOVIELENS RECOMMENDATION SYSTEM - STREAMLIT DEPLOYMENT
================================================================================

Deployment Application for MovieLens Recommendation System

Features:
---------
1. User-based recommendations
2. Movie search and similar movies
3. Business insights dashboard
4. Model performance comparison
5. Interactive filters

Author: NLC
Date: December 2025
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    padding: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.recommendation-box {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA & MODEL LOADING
# ============================================================================

@st.cache_data
def load_data(base_path):
    """Load processed data"""
    try:
        movies = pd.read_csv(os.path.join(base_path, 'movies_processed.csv'))
        ratings = pd.read_csv(os.path.join(base_path, 'ratings_processed.csv'))
        user_stats = pd.read_csv(os.path.join(base_path, 'user_statistics.csv'))
        genre_stats = pd.read_csv(os.path.join(base_path, 'genre_statistics.csv'))

        return movies, ratings, user_stats, genre_stats
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

@st.cache_resource
def load_models(models_path):
    """Load trained models"""
    try:
        baseline = joblib.load(os.path.join(models_path, 'baseline_popularity.pkl'))
        best_model = joblib.load(os.path.join(models_path, 'random_forest_tuned.pkl'))

        return baseline, best_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Set paths (adjust these for your environment)
BASE_PATH = "C:/Users/User/Downloads/ml-latest/ml-latest/output"
MODELS_PATH = os.path.join(BASE_PATH, 'models')
PLOTS_PATH = os.path.join(BASE_PATH, 'plots')

# Load data and models
movies, ratings, user_stats, genre_stats = load_data(BASE_PATH)
baseline_model, best_model = load_models(MODELS_PATH)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("# 🎬 MovieLens Recommender")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Navigate",
    ["🏠 Home", "👤 User Recommendations", "🔍 Movie Search",
     "📊 Business Insights", "🤖 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This recommendation system uses advanced machine learning "
    "to provide personalized movie recommendations based on "
    "the MovieLens dataset."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Statistics")
if movies is not None and ratings is not None:
    st.sidebar.metric("Total Movies", f"{len(movies):,}")
    st.sidebar.metric("Total Ratings", f"{len(ratings):,}")
    st.sidebar.metric("Total Users", f"{ratings['userId'].nunique():,}")

# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    st.markdown('<p class="main-header">🎬 MovieLens Recommendation System</p>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the MovieLens Recommender!

    This application provides:
    - 🎯 Personalized movie recommendations
    - 🔍 Similar movie discovery
    - 📊 Business insights and analytics
    - 🤖 Advanced ML model comparisons

    **Choose a page from the sidebar to get started!**
    """)

    # Display key metrics
    st.markdown("---")
    st.markdown("## 📈 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    if movies is not None and ratings is not None:
        with col1:
            st.metric(
                "Total Movies",
                f"{len(movies):,}",
                help="Total number of movies in the database"
            )

        with col2:
            st.metric(
                "Total Ratings",
                f"{len(ratings):,}",
                help="Total number of user ratings"
            )

        with col3:
            st.metric(
                "Active Users",
                f"{ratings['userId'].nunique():,}",
                help="Total number of users who rated movies"
            )

        with col4:
            avg_rating = ratings['rating'].mean()
            st.metric(
                "Average Rating",
                f"{avg_rating:.2f} ⭐",
                help="Average rating across all movies"
            )

    st.markdown("---")

    # Top rated movies preview
    if baseline_model is not None:
        st.markdown("## 🏆 Top Rated Movies")

        top_movies = baseline_model['movie_scores'].head(10)

        for idx, row in top_movies.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"Genres: {row['genres']}")

                with col2:
                    st.metric("Rating", f"{row['avg_rating']:.2f} ⭐")

                with col3:
                    st.metric("Votes", f"{row['vote_count']:,}")

                st.markdown("---")

# ============================================================================
# PAGE: USER RECOMMENDATIONS
# ============================================================================

elif page == "👤 User Recommendations":
    st.markdown('<p class="main-header">👤 Personalized Recommendations</p>',
                unsafe_allow_html=True)

    if movies is None or ratings is None or baseline_model is None:
        st.error("⚠️ Data or models not loaded. Please check your paths.")
    else:
        # User selection
        col1, col2 = st.columns([2, 1])

        with col1:
            user_id = st.selectbox(
                "Select User ID",
                options=sorted(ratings['userId'].unique()[:1000]),  # First 1000 users for demo
                help="Choose a user to see their personalized recommendations"
            )

        with col2:
            n_recommendations = st.slider(
                "Number of Recommendations",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )

        # Genre filter
        genres_list = ['All Genres'] + sorted(
            g for sublist in movies['genres'].str.split('|').dropna()
            for g in sublist if g != '(no genres listed)'
        )
        genre_filter = st.selectbox("Filter by Genre (optional)", genres_list)

        if st.button("🎯 Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                # Get user's watched movies
                user_ratings = ratings[ratings['userId'] == user_id]
                watched_movies = set(user_ratings['movieId'])

                # Get recommendations (movies not watched)
                recommendations = baseline_model['movie_scores'][
                    ~baseline_model['movie_scores']['movieId'].isin(watched_movies)
                ].copy()

                # Apply genre filter
                if genre_filter != 'All Genres':
                    recommendations = recommendations[
                        recommendations['genres'].str.contains(genre_filter, na=False)
                    ]

                recommendations = recommendations.head(n_recommendations)

                # Display user profile
                st.markdown("---")
                st.markdown("### 📊 User Profile")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Movies Rated", len(user_ratings))

                with col2:
                    st.metric("Average Rating", f"{user_ratings['rating'].mean():.2f} ⭐")

                with col3:
                    st.metric("Rating Std Dev", f"{user_ratings['rating'].std():.2f}")

                with col4:
                    rating_bias = "Generous" if user_ratings['rating'].mean() > 3.5 else \
                                 "Harsh" if user_ratings['rating'].mean() < 2.5 else "Moderate"
                    st.metric("Rating Bias", rating_bias)

                # Display recommendations
                st.markdown("---")
                st.markdown(f"### 🎬 Top {len(recommendations)} Recommendations")

                for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                    with st.expander(f"{idx}. {row['title']} - {row['weighted_score']:.2f} ⭐"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**Genres:** {row['genres']}")
                            release_year = row.get('release_year', np.nan)
                            if pd.notna(release_year):
                                st.markdown(f"**Year:** {int(release_year)}")

                        with col2:
                            st.metric("Score", f"{row['weighted_score']:.2f}")
                            st.metric("Avg Rating", f"{row['avg_rating']:.2f}")
                            st.metric("Votes", f"{row['vote_count']:,}")

# ============================================================================
# PAGE: MOVIE SEARCH
# ============================================================================

elif page == "🔍 Movie Search":
    st.markdown('<p class="main-header">🔍 Movie Search & Discovery</p>',
                unsafe_allow_html=True)

    if movies is None:
        st.error("⚠️ Movies data not loaded.")
    else:
        # Search functionality
        search_query = st.text_input(
            "🔎 Search for a movie",
            placeholder="Enter movie title...",
            help="Search by movie title"
        )

        if search_query:
            # Search movies
            search_results = movies[
                movies['title'].str.contains(search_query, case=False, na=False)
            ].head(20)

            if len(search_results) > 0:
                st.markdown(f"### Found {len(search_results)} movies")

                for _, movie in search_results.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])

                        with col1:
                            st.markdown(f"**{movie['title']}**")
                            st.caption(f"Genres: {movie['genres']}")

                        with col2:
                            release_year = movie.get('release_year', np.nan)
                            if pd.notna(release_year):
                                st.caption(f"Year: {int(release_year)}")

                        with col3:
                            if baseline_model is not None:
                                movie_score = baseline_model['movie_scores'][
                                    baseline_model['movie_scores']['movieId'] == movie['movieId']
                                ]
                                if len(movie_score) > 0:
                                    st.metric("Score", f"{movie_score.iloc[0]['weighted_score']:.2f}")

                        st.markdown("---")
            else:
                st.warning("No movies found. Try a different search term.")

        # Browse by genre
        st.markdown("---")
        st.markdown("### 🎭 Browse by Genre")

        genres_list = sorted(
            g for sublist in movies['genres'].str.split('|').dropna()
            for g in sublist if g != '(no genres listed)'
        )

        selected_genre = st.selectbox("Select a genre", genres_list)

        if selected_genre:
            genre_movies = movies[
                movies['genres'].str.contains(selected_genre, na=False)
            ]

            if baseline_model is not None:
                genre_movies = genre_movies.merge(
                    baseline_model['movie_scores'][['movieId', 'weighted_score', 'avg_rating', 'vote_count']],
                    on='movieId',
                    how='left'
                ).sort_values('weighted_score', ascending=False).head(20)

            st.markdown(f"#### Top 20 {selected_genre} Movies")

            for idx, movie in genre_movies.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.markdown(f"**{movie['title']}**")

                    with col2:
                        if 'weighted_score' in movie:
                            st.metric("Score", f"{movie['weighted_score']:.2f} ⭐")

                    with col3:
                        if 'vote_count' in movie:
                            st.metric("Votes", f"{int(movie['vote_count']):,}")

                    st.markdown("---")

# ============================================================================
# PAGE: BUSINESS INSIGHTS
# ============================================================================

elif page == "📊 Business Insights":
    st.markdown('<p class="main-header">📊 Business Insights Dashboard</p>',
                unsafe_allow_html=True)

    if genre_stats is not None:
        # Genre performance
        st.markdown("### 🎬 Genre Performance")

        fig = px.bar(
            genre_stats.sort_values('avg_rating', ascending=False).head(15),
            x='avg_rating',
            y='genre',
            orientation='h',
            title='Top 15 Genres by Average Rating',
            labels={'avg_rating': 'Average Rating', 'genre': 'Genre'},
            color='avg_rating',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Genre popularity
        st.markdown("---")

        fig2 = px.bar(
            genre_stats.sort_values('n_ratings', ascending=False).head(15),
            x='n_ratings',
            y='genre',
            orientation='h',
            title='Top 15 Most Popular Genres',
            labels={'n_ratings': 'Number of Ratings', 'genre': 'Genre'},
            color='n_ratings',
            color_continuous_scale='Blues'
        )
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)

    # Display saved visualizations
    st.markdown("---")
    st.markdown("### 📈 Analytics Visualizations")

    viz_files = [
        'cohort_retention_heatmap.png',
        'business_insights_dashboard.png',
        'genre_performance_analysis.png',
        'hidden_patterns_analysis.png'
    ]

    for viz_file in viz_files:
        viz_path = os.path.join(PLOTS_PATH, viz_file)
        if os.path.exists(viz_path):
            st.markdown(f"#### {viz_file.replace('_', ' ').replace('.png', '').title()}")
            image = Image.open(viz_path)
            st.image(image, use_column_width=True)
            st.markdown("---")

# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================

elif page == "🤖 Model Performance":
    st.markdown('<p class="main-header">🤖 Model Performance Comparison</p>',
                unsafe_allow_html=True)

    # Load model comparison
    comparison_path = os.path.join(BASE_PATH, 'reports', 'model_comparison.csv')

    if os.path.exists(comparison_path):
        model_comparison = pd.read_csv(comparison_path)

        st.markdown("### 📊 Model Comparison Table")
        st.dataframe(
            model_comparison.style.highlight_min(
                subset=['Test RMSE', 'Test MAE'],
                color='lightgreen'
            ).highlight_max(
                subset=['Test R²'],
                color='lightgreen'
            ),
            use_container_width=True
        )

        # RMSE comparison chart
        st.markdown("---")
        st.markdown("### 📈 RMSE Comparison")

        fig = px.bar(
            model_comparison.sort_values('Test RMSE'),
            x='Test RMSE',
            y='Model',
            orientation='h',
            title='Model Comparison by RMSE (Lower is Better)',
            labels={'Test RMSE': 'RMSE', 'Model': 'Model'},
            color='Test RMSE',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # R² comparison chart
        st.markdown("---")
        st.markdown("### 📈 R² Comparison")

        fig2 = px.bar(
            model_comparison.sort_values('Test R²', ascending=False),
            x='Test R²',
            y='Model',
            orientation='h',
            title='Model Comparison by R² (Higher is Better)',
            labels={'Test R²': 'R²', 'Model': 'Model'},
            color='Test R²',
            color_continuous_scale='RdYlGn'
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)

    # Display model comparison visualization
    st.markdown("---")

    comparison_viz = os.path.join(PLOTS_PATH, 'model_comparison.png')
    if os.path.exists(comparison_viz):
        st.markdown("### 🎯 Detailed Model Analysis")
        image = Image.open(comparison_viz)
        st.image(image, use_column_width=True)

    # Error analysis
    error_viz = os.path.join(PLOTS_PATH, 'error_analysis.png')
    if os.path.exists(error_viz):
        st.markdown("---")
        st.markdown("### 🔍 Error Analysis")
        image = Image.open(error_viz)
        st.image(image, use_column_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>MovieLens Recommendation System</strong></p>
    <p>Built with Streamlit • Powered by Machine Learning</p>
    <p>ML Project • December 2025</p>
</div>
""", unsafe_allow_html=True)
MODELS_PATH = os.path.join(BASE_PATH, "models")
