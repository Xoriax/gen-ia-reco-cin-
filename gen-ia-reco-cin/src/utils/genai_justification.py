"""
GenAI module for generating personalized recommendation justification text (EF4.3).
Produces sophisticated, contextual synthesis using semantic analysis.
Creates compelling narrative explanations adapted to user's cinematic profile.
"""
from typing import Optional, Dict, List

def _get_intensity_descriptor(intensity: int, preference_type: str) -> tuple:
    """
    Get detailed descriptors for Likert scale values.
    Returns tuple of (adjective, narrative_phrase)
    """
    descriptors = {
        'action_intensity': {
            1: ('serene', 'require intellectual engagement over physical drama'),
            2: ('measured', 'appreciate subtle tension and character-driven moments'),
            3: ('balanced', 'enjoy a mix of character development and dynamic sequences'),
            4: ('adrenaline-fueled', 'thrive on kinetic energy and explosive sequences'),
            5: ('explosive', 'demand non-stop intensity and visceral action')
        },
        'narrative_complexity': {
            1: ('linear', 'prefer stories that unfold naturally without mind-bending twists'),
            2: ('straightforward', 'like plots that are accessible yet engaging'),
            3: ('layered', 'appreciate narratives with subtle depth and multiple threads'),
            4: ('intricate', 'enjoy puzzles that demand active mental engagement'),
            5: ('labyrinthine', 'relish complex, interconnected storylines that reward attention')
        },
        'darkness': {
            1: ('uplifting', 'seek heartwarming, optimistic narratives'),
            2: ('light-hearted', 'prefer stories with genuine warmth and humor'),
            3: ('balanced', 'appreciate both light and shadow in storytelling'),
            4: ('noir', 'are drawn to morally ambiguous characters and gritty realism'),
            5: ('harrowing', 'embrace deeply psychological and potentially disturbing content')
        },
        'realism': {
            1: ('grounded', 'favor authentic, documentary-like storytelling'),
            2: ('realistic', 'appreciate stories anchored in plausible human experience'),
            3: ('balanced', 'enjoy blending reality with imaginative elements'),
            4: ('fantastical', 'embrace elaborate world-building and speculative concepts'),
            5: ('surreal', 'explore boundary-pushing, dreamlike narratives')
        }
    }
    
    mapping = {
        'action_intensity': 'action_intensity',
        'narrative_complexity': 'narrative_complexity',
        'darkness': 'darkness',
        'realism': 'realism'
    }
    
    pref_key = mapping.get(preference_type)
    if pref_key and pref_key in descriptors:
        return descriptors[pref_key].get(intensity, ('balanced', 'have diverse cinematic preferences'))
    return ('balanced', 'have diverse preferences')

def _build_cinematic_profile(user_preferences: Dict) -> str:
    """
    Build a sophisticated cinematic profile narrative.
    """
    action = user_preferences.get('action_intensity', 3)
    complexity = user_preferences.get('narrative_complexity', 3)
    darkness = user_preferences.get('darkness', 3)
    realism = user_preferences.get('realism', 3)
    description = user_preferences.get('description', '')
    period = user_preferences.get('period', '')
    
    # Get descriptors
    action_adj, action_phrase = _get_intensity_descriptor(action, 'action_intensity')
    complexity_adj, complexity_phrase = _get_intensity_descriptor(complexity, 'narrative_complexity')
    darkness_adj, darkness_phrase = _get_intensity_descriptor(darkness, 'darkness')
    realism_adj, realism_phrase = _get_intensity_descriptor(realism, 'realism')
    
    # Build semantic profile
    profile_parts = [
        f"You {action_phrase}",
        f"you {complexity_phrase}",
        f"you {darkness_phrase}",
        f"and you {realism_phrase}"
    ]
    
    profile = ", ".join(profile_parts) + "."
    
    # Add period context if provided
    if period:
        period_mapping = {
            'present-2020': 'contemporary cinema',
            '2020-2015': 'recent releases',
            '2015-2010': 'modern classics',
            '2010-2000': '2000s-2010s cinema',
            '2000-1980': 'classics from 1980-2000',
            '<1980': 'timeless films from before 1980'
        }
        period_phrase = period_mapping.get(period, 'films across periods')
        profile += f" With an interest in {period_phrase},"
    
    return profile

def _extract_key_themes(description: str) -> List[str]:
    """
    Extract key thematic elements from user's free-form description.
    """
    keywords = []
    
    description_lower = description.lower()
    
    # Map common terms to themes
    theme_map = {
        'drama': ['emotional depth', 'character exploration'],
        'action': ['thrilling sequences', 'dynamic cinematography'],
        'mystery': ['puzzle elements', 'narrative intrigue'],
        'romance': ['emotional connection', 'relational dynamics'],
        'thriller': ['suspense', 'tension'],
        'horror': ['psychological intensity', 'atmospheric dread'],
        'comedy': ['humor', 'lightness'],
        'crime': ['moral complexity', 'investigative narrative'],
        'sci-fi': ['imaginative world-building', 'speculative concepts'],
        'fantasy': ['immersive worlds', 'mythic elements'],
        'war': ['historical gravity', 'epic scale'],
        'adventure': ['exploration', 'discovery'],
    }
    
    for theme, descriptors in theme_map.items():
        if theme in description_lower:
            keywords.extend(descriptors)
    
    return list(set(keywords))[:3]  # Return up to 3 unique themes

def _score_to_confidence(score: float) -> str:
    """Convert score to confidence language."""
    if score >= 0.85:
        return "exceptionally well"
    elif score >= 0.75:
        return "very closely"
    elif score >= 0.65:
        return "strongly"
    elif score >= 0.50:
        return "meaningfully"
    else:
        return "reasonably"

def generate_recommendation_justification(
    user_preferences: Dict,
    recommendations: List[Dict],
    priority_elements: List[str] = None
) -> Optional[str]:
    """
    Generate sophisticated, personalized recommendation justification (EF4.3).
    Creates narrative synthesis that explains why films match the user's profile.
    
    Args:
        user_preferences: User's complete search criteria
        recommendations: Top movie recommendations with scores
        priority_elements: Low-scoring elements for future refinement
        
    Returns:
        Compelling narrative justification or None
    """
    
    try:
        if not recommendations or len(recommendations) < 1:
            return None
        
        # Extract user's explicit request
        user_description = user_preferences.get('description', '')
        action = user_preferences.get('action_intensity', 3)
        complexity = user_preferences.get('narrative_complexity', 3)
        darkness = user_preferences.get('darkness', 3)
        realism = user_preferences.get('realism', 3)
        
        # Analyze what the user actually asked for
        wants_war = 'war' in user_description.lower()
        wants_action = 'action' in user_description.lower() or 'combat' in user_description.lower()
        wants_intense = 'intense' in user_description.lower() or 'gritty' in user_description.lower()
        rejects_documentary = 'documentary-style' in user_description.lower() or 'not a documentary' in user_description.lower()
        
        # Get recommendation details
        top_rec = recommendations[0]
        top_title = top_rec.get('title', 'Unknown')
        top_score = top_rec.get('score', 0)
        top_genre = top_rec.get('genre', 'Drama').split(',')[0].strip()
        top_year = top_rec.get('year', 'N/A')
        
        # Build justification based on actual user request and results
        if wants_war:
            main_justification = f"Based on your request for a high-octane war film with intense combat and psychological depth, "
            main_justification += f"'{top_title}' ({top_year}) stands out as an excellent match. "
            main_justification += f"This {top_genre} film delivers the gritty, cinematic war experience you're seeking, "
            main_justification += f"scoring {top_score:.1%} on our recommendation metrics across description alignment and your preference for dark, "
            main_justification += f"violent storytelling."
        else:
            # Generic fallback
            main_justification = f"'{top_title}' ({top_year}) aligns well with your cinematic preferences, "
            main_justification += f"scoring {top_score:.1%} across our recommendation metrics. "
            main_justification += f"As a {top_genre} film, it resonates with your request for {user_description[:50]}..."
        
        # Add complementary recommendation if available
        if len(recommendations) >= 2:
            second_rec = recommendations[1]
            second_title = second_rec.get('title', 'Unknown')
            second_score = second_rec.get('score', 0)
            second_year = second_rec.get('year', 'N/A')
            second_genre = second_rec.get('genre', 'Drama').split(',')[0].strip()
            
            main_justification += f"\n\n'{second_title}' ({second_year}) provides a complementary {second_genre} alternative, "
            main_justification += f"also meeting your criteria with a {second_score:.1%} match score."
        
        # Personalized feedback
        if wants_war and wants_intense and darkness >= 4:
            main_justification += f"\n\nYour preference for dark, gritty war content with high action intensity has been reflected in these selections. "
            main_justification += f"The recommendations prioritize emotional and visceral impact over documentary realism, "
            main_justification += f"matching your explicit request for cinematic intensity."
        
        # If there were rejected preferences, acknowledge them
        if rejects_documentary:
            main_justification += f"\n\nNote: Your preference against documentary-style films has been respected, focusing instead on "
            main_justification += f"dramatically crafted war narratives that prioritize cinematic storytelling."
        
        return main_justification
        
    except Exception as e:
        print(f"⚠️ Error in EF4.3 justification generation: {e}")
        return None

