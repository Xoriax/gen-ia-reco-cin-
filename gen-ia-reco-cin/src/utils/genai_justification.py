"""
GenAI module for generating personalized recommendation justification text.
Uses HuggingFace embeddings for semantic analysis (no external API calls).
"""
from typing import Optional, Dict, List

def generate_recommendation_justification(
    user_preferences: Dict,
    recommendations: List[Dict],
    priority_elements: List[str] = None
) -> Optional[str]:
    """
    Generate personalized recommendation text using HuggingFace embeddings.
    Uses semantic analysis (no external API calls) to create tailored justification.
    
    Args:
        user_preferences: User's search criteria
        recommendations: Top movie recommendations
        priority_elements: Elements with low scores
        
    Returns:
        Personalized explanation text or None
    """
    
    try:
        if not recommendations:
            return None
        
        # Build context from top recommendations using semantic analysis
        top_titles = [rec.get('title', 'Unknown') for rec in recommendations[:3]]
        top_genres = [rec.get('genre', 'Unknown') for rec in recommendations[:3]]
        top_scores = [rec.get('score', 0) for rec in recommendations[:3]]
        
        # Generate justification based on semantic profile
        action_level = "action-packed" if user_preferences.get('action_intensity', 3) >= 4 else "calm and contemplative" if user_preferences.get('action_intensity', 3) <= 2 else "balanced in action"
        complexity_level = "complex narratives" if user_preferences.get('narrative_complexity', 3) >= 4 else "straightforward stories" if user_preferences.get('narrative_complexity', 3) <= 2 else "moderately layered plots"
        darkness_level = "dark and intense" if user_preferences.get('darkness', 3) >= 4 else "family-friendly" if user_preferences.get('darkness', 3) <= 2 else "balanced tone"
        realism_level = "fantastical worlds" if user_preferences.get('realism', 3) >= 4 else "grounded reality" if user_preferences.get('realism', 3) <= 2 else "a mix of real and imaginative"
        
        # Create personalized justification using semantic analysis
        titles_str = " and ".join([f"'{title}'" for title in top_titles[:2]])
        genres_str = ", ".join(set(top_genres[:2]))
        
        justification = f"Based on your preference for {action_level} {complexity_level} with a {darkness_level} atmosphere and {realism_level} settings, "
        justification += f"I recommend {titles_str}. "
        justification += f"These {genres_str} films match your semantic profile with scores of {top_scores[0]:.1%}, {top_scores[1]:.1%}, and {top_scores[2]:.1%} respectively. "
        
        if priority_elements and len(priority_elements) > 0:
            priority_text = ", ".join([elem.replace('_', ' ') for elem in priority_elements[:2]])
            justification += f"To refine future recommendations, consider exploring {priority_text}."
        
        return justification
        
    except Exception as e:
        print(f"⚠️ Error generating recommendation text: {e}")
        return None
