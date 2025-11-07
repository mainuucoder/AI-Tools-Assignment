import spacy
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_sm")

# Create a list of known brands and products (you can expand anytime)
brand_list = ["Apple", "Samsung", "Sony", "Nike", "Adidas", "Lenovo", "Dell", "Canon", "LG", "Microsoft"]
product_list = ["headphones", "laptop", "phone", "camera", "shoes", "watch", "tablet", "keyboard", "monitor", "speaker"]

# Add rules to spaCy
# Correct way to add EntityRuler: add by string name and get the returned component
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = []

for brand in brand_list:
    patterns.append({"label": "ORG", "pattern": brand})

for product in product_list:
    patterns.append({"label": "PRODUCT", "pattern": product})

ruler.add_patterns(patterns)

print("✅ Custom product/brand recognition added.")
def extract_entities(text):
    doc = nlp(text)
    brands = []
    products = []
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            brands.append(ent.text)
        elif ent.label_ == "PRODUCT":
            products.append(ent.text)
    
    return brands, products

df["Brands"], df["Products"] = zip(*df["review"].apply(extract_entities))
from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['review'].apply(get_sentiment)

print("✅ Sentiment column re-added to DataFrame!")
sample = df.sample(10, random_state=42)

for i, row in sample.iterrows():
    print("📝 Review:", row["review"])
    print("🏷️ Brands detected:", row["Brands"])
    print("📦 Products detected:", row["Products"])
    print("💭 Sentiment result:", row["Sentiment"])
    print("-" * 80)
# Summary of how many reviews are positive, negative, or neutral
df["Sentiment"].value_counts()
