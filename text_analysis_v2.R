#main outputs
#absa_results: Automatically extracted attributes and their sentiment predictions.
#grouped_aspects: Semantically similar attributes organized into clusters.
#absa_clustered: Consumer comments combined with their attribute groups and sentiment predictions.
#Sentiment word clouds and a comparison bar chart.

#we will use 3 models
#spaCy	Discover nouns and descriptive adjectives missed by RoBERTa
#RoBERTa	Identify additional aspect terms
#BART	Predict sentiment toward each extracted attribute
#MiniLM	Group semantically similar attributes

# install.packages(c(
#   "tidyverse",
#   "reticulate",
#   "ggwordcloud"
# ))

library(tidyverse)
library(reticulate)
library(ggwordcloud)

#configure a Python environment in r
reticulate::install_python()
reticulate::virtualenv_create("absa_env")

use_python(
  "C:/Users/despi/Envs/absa_env/Scripts/python.exe",
  required = TRUE
)

py_config()
py_install(
  c(
    "spacy",
    "torch",
    "transformers",
    "sentence-transformers",
    "pandas"
  ),
  envname = "absa_env",
  pip = TRUE
)

#install the spaCy English model
virtualenv_install(
  envname = "absa_env",
  packages = paste0(
    "https://github.com/explosion/spacy-models/releases/",
    "download/en_core_web_sm-3.8.0/",
    "en_core_web_sm-3.8.0-py3-none-any.whl"
  )
)

py_run_string("
import spacy
nlp = spacy.load('en_core_web_sm')
print('spaCy loaded successfully')
")

#data
feedback <- tibble(
  id = 1:10,
  comment = c(
    "I loved the creamy texture and sweet flavor",
    "Too sweet for my taste",
    "The texture was smooth and pleasant",
    "I liked the flavor but it was too thick",
    "It tasted artificial",
    "Great creamy texture and natural flavor",
    "Too sour and not sweet enough",
    "I enjoyed the smooth texture",
    "The flavor was good but too sweet",
    "I disliked the artificial aftertaste"
  )
)
print(feedback)

#extract attributes using spaCy and RoBERTa
#load the models
py_run_string("
import spacy
from transformers import pipeline

nlp = spacy.load('en_core_web_sm')

absa_model = pipeline(
    task='token-classification',
    model='gauneg/roberta-base-absa-ate-sentiment',
    aggregation_strategy='simple',
    device=-1
)
")

#define the extraction function
py_run_string("
import pandas as pd

generic_opinions = {
    'good', 'great', 'bad', 'excellent',
    'pleasant', 'nice', 'wonderful',
    'terrible', 'amazing', 'awful'
}

def extract_all_aspects(ids, comments):

    records = []

    for consumer_id, comment in zip(ids, comments):

        doc = nlp(comment)
        found = {}

        # Extract nouns and descriptive adjectives
        for token in doc:

            term = token.lemma_.lower().strip()

            is_noun = token.pos_ == 'NOUN'
            is_descriptive = token.pos_ == 'ADJ'

            if is_noun or is_descriptive:

                if (
                    term not in generic_opinions
                    and not token.is_stop
                    and token.is_alpha
                ):
                    found[term] = 'spaCy'

        # Extract additional aspects using RoBERTa
        predictions = absa_model(comment)

        for item in predictions:

            term = item['word'].strip().lower()

            if term:

                if term in found:
                    found[term] = 'Both'
                else:
                    found[term] = 'RoBERTa'

        # Save extracted aspects
        for term, source in found.items():

            records.append({
                'id': int(consumer_id),
                'comment': comment,
                'aspect': term,
                'source': source
            })

    return pd.DataFrame(
        records,
        columns=[
            'id',
            'comment',
            'aspect',
            'source'
        ]
    )
")

#run the extraction
aspect_candidates <- py_to_r(
  py$extract_all_aspects(
    as.list(feedback$id),
    as.list(feedback$comment)
  )
) %>%
  as_tibble()

print(aspect_candidates, n = Inf)

aspect_candidates %>%
  count(aspect, sort = TRUE) %>%
  print(n = Inf)

py_has_attr(py, "classify_sentiment")
py_has_attr(py, "sentiment_model")

#load a context-aware classifier
library(reticulate)

py_run_string("
from transformers import pipeline

sentiment_model = pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli',
    device=-1
)
")

#create a sentiment classification function
py_run_string("
def classify_sentiment(comment, aspect):

    doc = nlp(comment)
    target = aspect.lower().strip()

    for token in doc:

        if token.lemma_.lower() == target:

            before = [
                t.lower_
                for t in doc[max(0, token.i - 2):token.i]
            ]

            after = [
                t.lower_
                for t in doc[
                    token.i + 1:
                    min(len(doc), token.i + 3)
                ]
            ]

            if 'too' in before:
                return {
                    'sentiment': 'Negative',
                    'confidence': None,
                    'method': 'Intensity rule'
                }

            if 'not' in before and 'enough' in after:
                return {
                    'sentiment': 'Negative',
                    'confidence': None,
                    'method': 'Intensity rule'
                }

    labels = [
        'satisfied with this attribute',
        'dissatisfied with this attribute',
        'neutral about this attribute'
    ]

    result = sentiment_model(
        f'Consumer comment: {comment}. Attribute: {aspect}.',
        candidate_labels=labels,
        hypothesis_template='The consumer is {}.'
    )

    mapping = {
        labels[0]: 'Positive',
        labels[1]: 'Negative',
        labels[2]: 'Neutral'
    }

    return {
        'sentiment': mapping[result['labels'][0]],
        'confidence': float(result['scores'][0]),
        'method': 'BART'
    }
")
py$classify_sentiment(
  "Too sweet for my taste",
  "sweet"
)

py$classify_sentiment(
  "Too sour and not sweet enough",
  "sweet"
)

py$classify_sentiment(
  "I liked the flavor but it was too thick",
  "thick"
)

#apply sentiment analysis to all attributes
library(reticulate)

py_run_string("
import pandas as pd

def analyze_candidates(ids, comments, aspects, sources):

    output = []

    for consumer_id, comment, aspect, source in zip(
        ids, comments, aspects, sources
    ):

        result = classify_sentiment(comment, aspect)

        output.append({
            'id': int(consumer_id),
            'comment': comment,
            'aspect': aspect,
            'source': source,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'method': result['method']
        })

    return pd.DataFrame(output)
")

py_to_r(
  py$classify_sentiment(
    "I liked the flavor but it was too thick",
    "thick"
  )
)

py_to_r(
  py$classify_sentiment(
    "I liked the flavor but it was too thick",
    "thick"
  )
)

#call it from R and display the results
absa_results <- py$analyze_candidates(
  as.list(aspect_candidates$id),
  as.list(aspect_candidates$comment),
  as.list(aspect_candidates$aspect),
  as.list(aspect_candidates$source)
) %>%
  as_tibble()

print(absa_results, n = Inf, width = Inf)

absa_results %>%
  select(
    id,
    aspect,
    sentiment,
    confidence,
    method
  ) %>%
  arrange(id) %>%
  print(n = Inf)

#check the problematic consumer responses
absa_results %>%
  filter(id %in% c(2, 4, 7, 9, 10)) %>%
  select(
    id,
    comment,
    aspect,
    sentiment,
    method
  ) %>%
  print(n = Inf, width = Inf)

#prepare the attributes for semantic clustering
aspect_data <- absa_results %>%
  mutate(
    aspect = str_to_lower(
      str_squish(aspect)
    ),
    sentiment = str_to_title(sentiment)
  ) %>%
  filter(
    !is.na(aspect),
    aspect != ""
  )

unique_aspects <- aspect_data %>%
  distinct(aspect) %>%
  arrange(aspect)

print(unique_aspects)
nrow(unique_aspects)

#generate semantic embeddings
py_run_string("
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2'
)
")

aspect_embeddings <- py_to_r(
  py$embedding_model$encode(
    as.list(unique_aspects$aspect),
    normalize_embeddings = TRUE
  )
)

aspect_embeddings <- as.matrix(aspect_embeddings)

stopifnot(
  nrow(aspect_embeddings) == nrow(unique_aspects)
)

rownames(aspect_embeddings) <- unique_aspects$aspect

print(dim(aspect_embeddings))

#each aspect is now represented by a numerical embedding

#calculate semantic similarity
similarity_matrix <- aspect_embeddings %*%
  t(aspect_embeddings)
print(round(similarity_matrix, 3))

#calculate the distance matrix
distance_values <- 1 - similarity_matrix
distance_values[distance_values < 0] <- 0
diag(distance_values) <- 0
distance_matrix <- as.dist(distance_values)

#cluster similar attributes
if (nrow(unique_aspects) > 1) {
  hc <- hclust(
    distance_matrix,
    method = "complete"
  )
  plot(
    hc,
    main = "Semantic Clustering of Yogurt Attributes",
    xlab = "Attributes",
    sub = "",
    hang = -1
  )
  unique_aspects$cluster <- cutree(
    hc,
    h = 0.35
  )
} else {
  unique_aspects$cluster <- 1L
}

#name the clusters
cluster_labels <- unique_aspects %>%
  group_by(cluster) %>%
  group_modify(~ {
    
    members <- .x$aspect
    
    sim <- similarity_matrix[
      members,
      members,
      drop = FALSE
    ]
    
    representative <- members[
      which.max(rowMeans(sim))
    ]
    
    tibble(
      cluster_label = representative,
      n_terms = length(members)
    )
    
  }) %>%
  ungroup()

#combine the labels with the original attributes
grouped_aspects <- unique_aspects %>%
  left_join(
    cluster_labels,
    by = "cluster"
  ) %>%
  arrange(cluster, aspect)

print(grouped_aspects, n = Inf)

#combine clusters with sentiment results
absa_clustered <- aspect_data %>%
  left_join(
    grouped_aspects %>%
      select(
        aspect,
        cluster,
        cluster_label
      ),
    by = "aspect"
  )

print(absa_clustered, n = Inf)

#summarize sentiment by attribute group
cluster_summary <- absa_clustered %>%
  filter(
    sentiment %in% c(
      "Positive",
      "Negative",
      "Neutral"
    )
  ) %>%
  distinct(
    id,
    cluster,
    cluster_label,
    sentiment
  ) %>%
  count(
    cluster,
    cluster_label,
    sentiment,
    name = "frequency"
  )

print(cluster_summary)

#create the sentiment word cloud
set.seed(123)

ggplot(
  cluster_summary,
  aes(
    label = cluster_label,
    size = frequency,
    color = sentiment
  )
) +
  geom_text_wordcloud(
    rm_outside = FALSE
  ) +
  facet_wrap(
    ~ sentiment,
    scales = "free"
  ) +
  scale_size_area(max_size = 22) +
  scale_color_manual(
    values = c(
      "Positive" = "darkgreen",
      "Negative" = "red3",
      "Neutral" = "gray50"
    )
  ) +
  labs(
    title = "Sentiment Word Clouds",
    subtitle = "Automatically Extracted and Grouped Attributes"
  ) +
  theme_void() +
  theme(
    legend.position = "none",
    strip.text = element_text(
      size = 13,
      face = "bold"
    )
  )

#create a sentiment bar chart
ggplot(
  cluster_summary,
  aes(
    x = cluster_label,
    y = frequency,
    fill = sentiment
  )
) +
  geom_col(position = "dodge") +
  coord_flip() +
  scale_fill_manual(
    values = c(
      "Positive" = "darkgreen",
      "Negative" = "red3",
      "Neutral" = "gray50"
    )
  ) +
  labs(
    title = "Consumer Sentiment by Attribute Group",
    x = "Attribute Group",
    y = "Number of Consumers",
    fill = "Sentiment"
  ) +
  theme_minimal()

#validate results
validation_table <- absa_clustered %>%
  select(
    id,
    comment,
    aspect,
    sentiment,
    method,
    cluster_label
  ) %>%
  arrange(id, aspect)

print(validation_table, n = Inf, width = Inf)

#identify low confidence model predictions
absa_clustered %>%
  filter(
    method == "BART",
    !is.na(confidence),
    confidence < 0.65
  ) %>%
  select(
    id,
    aspect,
    sentiment,
    confidence,
    comment
  )

#inspect how the attributes were clustered
grouped_aspects %>%
  group_by(cluster, cluster_label) %>%
  summarise(
    attributes = paste(aspect, collapse = ", "),
    .groups = "drop"
  ) %>%
  print(n = Inf)

#export the final results
write.csv(
  absa_clustered,
  "yogurt_absa_results.csv",
  row.names = FALSE
)

write.csv(
  grouped_aspects,
  "yogurt_semantic_clusters.csv",
  row.names = FALSE
)

write.csv(
  cluster_summary,
  "yogurt_sentiment_summary.csv",
  row.names = FALSE
)

