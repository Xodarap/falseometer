import arxiv

# Construct the default API client.
client = arxiv.Client()

terms = ["curiosity", "human parity", ]

# Search for the 10 most recent articles matching the keyword "quantum."
search = arxiv.Search(
  query = f'all:cs.lg AND abs:{terms[0]}',
  max_results = 10,
  sort_by = arxiv.SortCriterion.SubmittedDate
)

results = client.results(search)
# `results` is a generator; you can iterate over its elements one by one...
for r in client.results(search):
  r.download_pdf("files/linguistic/", r.title + ".pdf")