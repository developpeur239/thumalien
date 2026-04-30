from atproto import Client
from supabase import create_client
from dotenv import load_dotenv
import os, time
from datetime import datetime

load_dotenv()

class BlueskyCollector:
    def __init__(self):
        # Connexion Bluesky
        self.client = Client()
        self.client.login(
            os.getenv("BLUESKY_HANDLE"),
            os.getenv("BLUESKY_PASSWORD")
        )
        print("✅ Connecté à Bluesky")

        # Connexion Supabase
        self.db = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        print("✅ Connecté à Supabase")

    def collect_posts(self, query: str, limit: int = 100):
        """Collecte des posts sur un sujet donné"""
        posts = []
        
        try:
            response = self.client.app.bsky.feed.search_posts(
                params={"q": query, "limit": min(limit, 100)}
            )
            
            for post in response.posts:
                posts.append({
                    "id": post.uri,
                    "text": post.record.text,
                    "author": post.author.handle,
                    "created_at": post.record.created_at[:19].replace("T", " ") if post.record.created_at else None,
                    "likes": post.like_count or 0,
                    "reposts": post.repost_count or 0,
                    "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": query,
                    "label": None,
                    "fake_score": None
                })
                
        except Exception as e:
            print(f"❌ Erreur collecte : {e}")
        
        return posts

    def save_to_supabase(self, posts: list):
        """Sauvegarde directement dans Supabase"""
        if not posts:
            print("⚠️ Aucun post à sauvegarder")
            return
        
        try:
            # Upsert = insert ou update si déjà existant
            self.db.table("posts").upsert(posts).execute()
            print(f"💾 {len(posts)} posts sauvegardés dans Supabase")
        except Exception as e:
            print(f"❌ Erreur sauvegarde : {e}")


if __name__ == "__main__":
    collector = BlueskyCollector()
    
    queries = [
        "complot", "fake news", "désinformation",
        "vaccin danger", "5G", "élection fraude"
    ]
    
    all_posts = []
    for query in queries:
        print(f"🔍 Recherche : '{query}'")
        posts = collector.collect_posts(query, limit=50)
        all_posts.extend(posts)
        time.sleep(1)
    
    collector.save_to_supabase(all_posts)
    print(f"\n✅ Total collecté : {len(all_posts)} posts")