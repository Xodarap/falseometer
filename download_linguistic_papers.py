import arxiv

# Construct the default API client.
client = arxiv.Client()

terms = ["curiosity", "human parity", "deception", "fear", "comprehension", "disparate impact", "fair", "dream", "creative", '"thought vector"', '"consciousness prior"', '"thought process"', "forget"]

# Search for the 10 most recent articles matching the keyword "quantum."
search = arxiv.Search(
  query = f'all:cs.lg AND abs:{terms[12]}',
  max_results = 20,
  sort_by = arxiv.SortCriterion.SubmittedDate
)

results = client.results(search)
c = 0
for r in client.results(search):
  c+=1
  try:
    r.download_pdf("files/linguistic/", r.title + ".pdf")
  except:
    print(f"Error downloading {r.title}")
    continue