from sentence_transformers import SentenceTransformer, util

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode two sentences
majorProcesses = ["Base", "Walls", "Roof", "Electrical"]
baseList = ["Base framing erection", "Subfloor installation"]
wallList = ["Wall framing erection", "Siding barrier and installation", "Window fitting", "Interior hardware installation", "Wall insulation", "Interior painting", "Exterior painting"]
roofList = ["Roof framing erection", "Roof vapor barrier and tile installation", "Roofing insulation"]
electricalList = ["Building power installation", "Lighting installation",
                  "Low voltage electrical installation", "Solar panel system install", "Backup power system test"]


lists = [baseList, wallList, roofList, electricalList]


for process in majorProcesses:
    embeddings1 = model.encode(process, convert_to_tensor=True)
    for list in lists:
        lowestValue = 1
        for activity in list:
            embeddings2 = model.encode(activity, convert_to_tensor=True)

            # Calculate cosine similarity between the sentence embeddings
            cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
            val = cosine_similarity[0][0].item()
            if val < lowestValue:
                lowestValue = val
        print(process, list, " had the lowest value of: ", lowestValue)
        print()
        print()

# Calculate cosine similarity between the sentence embeddings
# cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

# print("Cosine Similarity:", cosine_similarity[0][0].item())
# print(type(cosine_similarity))

