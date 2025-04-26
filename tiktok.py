import os
import json
import random
import time
from datetime import datetime, timedelta

class TikTokAPIClient:
    """
    Simulated TikTok API client for Filter Bubble Analyzer
    """
    
    def __init__(self):
        """Initialize the API client"""
        self.data_dir = 'sample_data'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Topics for generating sample data
        self.topics = [
            "Elections", "Political_Figures", "Economic_Policy",
            "Social_Issues", "Healthcare", "Environment", 
            "International_Relations", "Constitutional_Issues",
            "General_Politics"
        ]
    
    def get_user_feed(self, username, limit=10):
        """Get a simulated TikTok feed"""
        # Check if we have cached data for this user
        cache_file = f"{self.data_dir}/{username}_feed.json"
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                print(f"Using cached feed data for {username}")
                return data[:limit]
            except Exception as e:
                print(f"Error reading cache: {str(e)}")
        
        # Generate sample data
        print(f"Generating sample feed data for {username}")
        data = self._generate_sample_feed(username)
        
        # Cache the data
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return data[:limit]
    
    def _generate_sample_feed(self, username):
        """Generate a simulated TikTok feed with enhanced diversity and realism"""
        videos = []
        now = datetime.now()
        
        # Determine user profile type and bias based on username
        if username.lower().endswith(('_left', '_liberal', '_progressive')):
            # Left-leaning user
            political_ratio = 0.7
            bias_weights = {'left': 0.7, 'center': 0.25, 'right': 0.05}
            topic_focus = random.sample(self.topics, 3)  # Focus on 3 random topics
        elif username.lower().endswith(('_right', '_conservative', '_republican')):
            # Right-leaning user
            political_ratio = 0.7
            bias_weights = {'left': 0.05, 'center': 0.25, 'right': 0.7}
            topic_focus = random.sample(self.topics, 3)  # Focus on 3 random topics
        elif username.lower().endswith(('_moderate', '_centrist', '_independent')):
            # Balanced user
            political_ratio = 0.5
            bias_weights = {'left': 0.33, 'center': 0.34, 'right': 0.33}
            topic_focus = random.sample(self.topics, 5)  # More diverse topics
        else:
            # Create a semi-random profile based on username hash
            username_hash = sum(ord(c) for c in username)
            random.seed(username_hash)  # Make consistent random choices for the same username
            
            political_ratio = random.uniform(0.3, 0.7)
            
            # Slightly biased distribution based on username
            left_weight = random.uniform(0.2, 0.5)
            right_weight = random.uniform(0.2, 0.5)
            center_weight = 1.0 - (left_weight + right_weight)
            
            bias_weights = {
                'left': left_weight,
                'center': center_weight,
                'right': right_weight
            }
            
            # Normalize weights to sum to 1
            total = sum(bias_weights.values())
            bias_weights = {k: v/total for k, v in bias_weights.items()}
            
            # Select 4 random topics to focus on
            topic_focus = random.sample(self.topics, 4)
        
        # Generate videos
        for i in range(100):  # Generate 100 videos
            # Determine if political
            is_political = random.random() < political_ratio
            
            if is_political:
                # Select bias based on weights
                bias = random.choices(
                    population=['left', 'center', 'right'],
                    weights=[bias_weights['left'], bias_weights['center'], bias_weights['right']],
                    k=1
                )[0]
                
                # Select topic - higher chance of being from focus topics
                if random.random() < 0.7 and topic_focus:
                    topic = random.choice(topic_focus)
                else:
                    topic = random.choice(self.topics)
                
                # Generate text based on bias and topic
                text = self._generate_political_text(bias, topic)
            else:
                # Generate non-political text
                text = self._generate_nonpolitical_text()
                bias = None
                topic = None
            
            # Generate engagement metrics - more sophisticated model
            # Base views depends on recency and randomness
            days_ago = random.randint(0, 30)
            recency_factor = 1.0 - (days_ago / 60.0)  # Recent videos get more views
            base_views = int(random.randint(1000, 100000) * recency_factor)
            
            # Engagement rates vary
            if is_political:
                # Political content often has higher engagement
                like_rate = random.uniform(0.15, 0.35)
                comment_rate = random.uniform(0.02, 0.08)
                share_rate = random.uniform(0.02, 0.15)
                
                # Controversial bias types (left/right) get more engagement than center
                if bias != "center":
                    like_rate *= 1.2
                    comment_rate *= 1.5
                    share_rate *= 1.3
            else:
                # Non-political content
                like_rate = random.uniform(0.1, 0.3)
                comment_rate = random.uniform(0.01, 0.05)
                share_rate = random.uniform(0.01, 0.1)
            
            likes = int(base_views * like_rate)
            comments = int(base_views * comment_rate)
            shares = int(base_views * share_rate)
            
            # Create timestamp (random time in last 30 days)
            timestamp = int((now - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))).timestamp())
            
            # Create video object
            video = {
                "video_id": f"{username}_{i}",
                "text": text,
                "views": base_views,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "create_time": timestamp,
                "_synthetic_data": {
                    "is_political": is_political,
                    "bias": bias,
                    "topic": topic
                }
            }
            
            videos.append(video)
        
        # Sort videos by create_time descending (newest first)
        videos.sort(key=lambda x: x["create_time"], reverse=True)
        
        return videos
    
    def _generate_political_text(self, bias, topic):
        """Generate sample political text based on bias and topic with more nuance"""
        # Templates by bias
        templates = {
            'left': [
                "We need to protect {issue} for everyone. #{topic} #progressive",
                "It's time for real change on {issue}. #{topic} #equality",
                "{issue} should be a right, not a privilege. #{topic} #fairness",
                "The latest policy on {issue} doesn't go far enough. #{topic} #reform",
                "We can do better on {issue} if we work together. #{topic} #community",
                "This is why {issue} matters to everyday Americans. #{topic} #progress",
                "The data on {issue} speaks for itself. #{topic} #science",
                "Why are we still debating {issue} in this day and age? #{topic} #forward",
                "Here's what they don't want you to know about {issue}. #{topic} #truth",
                "Our future depends on how we handle {issue} today. #{topic} #sustainability"
            ],
            'center': [
                "There are reasonable compromises on {issue} we should consider. #{topic} #balance",
                "Both sides have valid points about {issue}. #{topic} #perspective",
                "We need practical solutions for {issue}, not partisan ones. #{topic} #common_sense",
                "The debate on {issue} needs more nuance. #{topic} #moderation",
                "Finding middle ground on {issue} is essential. #{topic} #compromise",
                "Let's look at the facts on {issue} without the politics. #{topic} #truth",
                "Here's an objective analysis of {issue} you need to see. #{topic} #facts",
                "Is there a better way to approach {issue}? I think so. #{topic} #solutions",
                "The polarization around {issue} isn't helping anyone. #{topic} #unity",
                "Policy over politics when it comes to {issue}. #{topic} #results"
            ],
            'right': [
                "Government overreach on {issue} threatens our freedoms. #{topic} #liberty",
                "The free market offers the best solutions for {issue}. #{topic} #freedom",
                "Traditional values should guide our approach to {issue}. #{topic} #tradition",
                "Personal responsibility is key to solving {issue}. #{topic} #individual",
                "Regulations on {issue} are hurting economic growth. #{topic} #deregulation",
                "The founding fathers would have a lot to say about {issue}. #{topic} #constitution",
                "Hard-working Americans deserve better policies on {issue}. #{topic} #opportunity",
                "The media won't show you the truth about {issue}. #{topic} #reality",
                "Family values and {issue} go hand in hand. #{topic} #faith",
                "Why big government can't fix {issue} but you can. #{topic} #smallgov"
            ]
        }
        
        # Issues by topic
        issues = {
            "Elections": ["voting rights", "election security", "campaign finance", "ballot access", "voter ID", 
                         "electoral college", "ranked choice voting", "gerrymandering", "vote by mail", 
                         "automatic voter registration", "election day holiday"],
            
            "Political_Figures": ["presidential leadership", "congressional oversight", "political appointments", 
                                 "executive orders", "political scandals", "cabinet positions", "supreme court nominations",
                                 "congressional leadership", "presidential candidates", "gubernatorial races"],
            
            "Economic_Policy": ["taxes", "government spending", "job creation", "minimum wage", "trade policy",
                               "budget deficit", "wealth inequality", "corporate regulation", "universal basic income",
                               "federal reserve", "inflation", "monetary policy", "stimulus packages"],
            
            "Social_Issues": ["gun control", "abortion", "immigration", "LGBTQ rights", "racial equity",
                             "religious freedom", "free speech", "drug legalization", "capital punishment",
                             "homelessness", "disability rights", "education access", "food insecurity"],
            
            "Healthcare": ["healthcare access", "insurance coverage", "drug pricing", "medical research", "public health",
                          "medicare for all", "affordable care act", "telehealth", "mental health services",
                          "hospital funding", "healthcare worker conditions", "preventative care", "rural healthcare"],
            
            "Environment": ["climate change", "renewable energy", "conservation", "pollution", "environmental regulations",
                           "carbon tax", "fossil fuel subsidies", "clean water", "air quality", "recycling programs",
                           "national parks", "endangered species", "sustainable agriculture"],
            
            "International_Relations": ["foreign policy", "military intervention", "trade agreements", "diplomatic relations", 
                                       "international aid", "sanctions", "nuclear proliferation", "peacekeeping missions",
                                       "refugee policy", "military alliances", "foreign election interference"],
            
            "Constitutional_Issues": ["first amendment", "second amendment", "fourth amendment", "civil liberties", 
                                     "supreme court decisions", "executive privilege", "legislative powers",
                                     "states' rights", "constitutional interpretation", "checks and balances"],
            
            "General_Politics": ["political division", "media coverage", "legislative reform", "judicial nominations", 
                                "political participation", "bureaucracy", "transparency", "term limits",
                                "lobbying", "campaign reform", "public trust", "civic education"]
        }
        
        # Select template and issue
        template = random.choice(templates[bias])
        issue = random.choice(issues.get(topic, issues["General_Politics"]))
        topic_tag = topic.lower().replace('_', '')
        
        # Format the text
        return template.format(issue=issue, topic=topic_tag)
    
    def _generate_nonpolitical_text(self):
        """Generate sample non-political text with more creative variety"""
        templates = [
            "Check out this amazing {thing}! #{category} #trending",
            "My {relation} couldn't believe this {thing}! #{category} #funny",
            "The best way to {action} your {thing}. #{category} #tips",
            "POV: When your {relation} {action} your {thing}. #{category} #relatable",
            "This {thing} hack will change your life! #{category} #lifehack",
            "Day {number} of {action} my {thing}. #{category} #challenge",
            "Wait for it... 😂 #{category} #funny",
            "Trying this new {thing} trend. #{category} #newtrend",
            "Reply with your favorite {thing}! #{category} #community",
            "Duet this if you also {action} your {thing}! #{category} #duet",
            "I spent {number} hours making this {thing}. Worth it? #{category} #create",
            "Tell me you're a {adjective} person without telling me. #{category} #tellme",
            "They said I couldn't {action} a {thing}. Watch this! #{category} #prove",
            "This {thing} sound is everything 🔥 #{category} #sound",
            "{number} types of {thing} - which one are you? #{category} #types",
            "My {thing} routine that changed everything. #{category} #routine",
            "When the {thing} is just too {adjective}... #{category} #moment",
            "POV: It's 2010 and your {relation} is {action} the {thing}. #{category} #nostalgia",
            "Real talk about {thing} - no one's ready for this conversation. #{category} #realtalk",
            "I can't believe what happened with this {thing}! #{category} #unexpected"
        ]
        
        categories = [
            "dance", "comedy", "food", "beauty", "fitness", 
            "fashion", "pets", "music", "travel", "art",
            "gaming", "diy", "comedy", "cooking", "gardening",
            "reading", "film", "photography", "crafts", "wellness"
        ]
        
        things = [
            "recipe", "outfit", "routine", "hairstyle", "workout",
            "dog", "cat", "song", "vacation", "painting", "video",
            "trick", "dessert", "makeup", "hack", "app", "game",
            "house plant", "morning ritual", "book", "movie", "coffee",
            "selfcare", "playlist", "street food", "thrift find", "home decor",
            "life story", "collection", "gadget", "roadtrip", "skincare"
        ]
        
        actions = [
            "improve", "fix", "style", "customize", "organize",
            "clean", "prepare", "create", "discover", "transform",
            "share", "love", "miss", "find", "use", "make",
            "upgrade", "build", "design", "master", "learn", "perfect",
            "hack", "renovate", "streamline", "elevate", "simplify",
            "decorate", "repurpose", "showcase", "celebrate", "explore"
        ]
        
        relations = [
            "friend", "sister", "brother", "mom", "dad", 
            "roommate", "cousin", "teacher", "boss", "neighbor",
            "partner", "grandma", "grandpa", "classmate", "coworker",
            "cat", "dog", "pet", "barista", "hairdresser", "coach",
            "dentist", "uber driver", "landlord", "ex", "crush"
        ]
        
        adjectives = [
            "creative", "organized", "spontaneous", "thoughtful", "adventurous",
            "introverted", "extroverted", "analytical", "artistic", "athletic",
            "tech-savvy", "bookish", "foodie", "fashionable", "minimalist",
            "perfectionist", "anxious", "chill", "nostalgic", "quirky", "extra",
            "basic", "bougie", "outdoorsy", "sensitive", "dramatic", "indecisive"
        ]
        
        # Select random elements
        template = random.choice(templates)
        category = random.choice(categories)
        thing = random.choice(things)
        action = random.choice(actions)
        relation = random.choice(relations)
        adjective = random.choice(adjectives)
        number = random.randint(1, 100)
        
        # Format the text
        return template.format(
            category=category,
            thing=thing,
            action=action,
            relation=relation,
            adjective=adjective,
            number=number
        )