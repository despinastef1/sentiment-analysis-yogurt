#https://huggingface.co/models

library(tidyverse)
library(tidytext)
library(textdata)
library(sentimentr)
library(dplyr)
library(wordcloud2)

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


words <- feedback %>%
  unnest_tokens(word, comment) %>%  #tokenization
  anti_join(stop_words, by = "word")  #removes common words as the, and, was, I

print(words)

word_counts <- words %>%
  count(word, sort = TRUE)

print(word_counts)

wordcloud2(
  data = word_counts,
  size = 1,
  color = "random-dark",
  backgroundColor = "white"
)

word_counts %>%
  slice_max(n, n = 10) %>%
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Most Frequently Mentioned Words",
    x = "Word",
    y = "Frequency"
  ) +
  theme_minimal()

#identify themes using qualitative coding
#assume the themes are texture, sweetness, flavor, sourness, artificial taste
coded_feedback <- feedback %>%
  mutate(
    texture = str_detect(
      str_to_lower(comment),
      "texture|creamy|smooth|thick"
    ),
    sweetness = str_detect(
      str_to_lower(comment),
      "sweet"
    ),
    flavor = str_detect(
      str_to_lower(comment),
      "flavor|tasted|taste|aftertaste"
    ),
    sourness = str_detect(
      str_to_lower(comment),
      "sour"
    ),
    artificial = str_detect(
      str_to_lower(comment),
      "artificial"
    )
  )

print(coded_feedback)

#summarize theme frequencies
theme_counts <- coded_feedback %>%
  select(texture:artificial) %>%
  summarise(across(everything(), sum)) %>%
  pivot_longer(
    cols = everything(),
    names_to = "theme",
    values_to = "frequency"
  ) %>%
  arrange(desc(frequency))

print(theme_counts)

ggplot(
  theme_counts,
  aes(x = reorder(theme, frequency), y = frequency)
) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Consumer Feedback Themes",
    x = "Theme",
    y = "Number of Consumers"
  ) +
  theme_minimal()

#sentiment analysis
#apply a predefined sentiment dictionary
#bing sentiment lexicon classifies words as positive or negative

sentiment_words <- feedback %>%
  unnest_tokens(word, comment) %>%
  inner_join(
    get_sentiments("bing"),
    by = "word"
  )

print(sentiment_words)

#calculate a sentiment score for each consumer 
#as the difference between positive word count and negative word count

sentiment_scores <- sentiment_words %>%
  count(id, sentiment) %>%
  pivot_wider(
    names_from = sentiment,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(
    score = positive - negative
  )

feedback_results <- feedback %>%
  left_join(sentiment_scores, by = "id") %>%
  mutate(
    across(
      c(positive, negative, score),
      ~replace_na(.x, 0)
    )
  )

print(feedback_results)


#a better method for sentiment analysis is context-aware sentiment
#analyze the entire comment
results <- sentiment_by(
  text.var = feedback$comment,
  by = feedback$id
)

print(results)

#combine with original comments
feedback_sentiment <- feedback %>%
  left_join(
    results %>%
      rename(id = 1) %>%
      mutate(id = as.integer(as.character(id))) %>%
      select(id, ave_sentiment),
    by = "id"
  ) %>%
  rename(sentiment_score = ave_sentiment) %>%
  mutate(
    sentiment_class = case_when(
      sentiment_score > 0 ~ "Positive",
      sentiment_score < 0 ~ "Negative",
      TRUE ~ "Neutral"
    )
  )

print(feedback_sentiment)

#visualize the results
ggplot(
  feedback_sentiment,
  aes(
    x = factor(id),
    y = sentiment_score,
    fill = sentiment_class
  )
) +
  geom_col() +
  scale_fill_manual(
    values = c(
      "Positive" = "seagreen",
      "Negative" = "tomato",
      "Neutral" = "gray"
    )
  ) +
  labs(
    title = "Automated Consumer Sentiment Analysis",
    x = "Consumer ID",
    y = "Sentiment Score"
  ) +
  theme_minimal()

#still, this code cannot identify "too thick" as having a negative connotation
#implement aspect-based sentiment analysis
#we will use the pretrained DeBERTa ABSA model on Hugging Face

# install.packages(c(
#   "tidyverse",
#   "reticulate"
# ))

library(tidyverse)
library(reticulate)

#configure a Python environment in r
reticulate::install_python()

reticulate::virtualenv_create("absa_env")

reticulate::py_install(
  c("torch", "transformers", "sentencepiece"),
  envname = "absa_env",
  pip = TRUE
)


use_virtualenv("absa_env", required = TRUE)

#identify product attributes
aspects <- tibble(
  aspect = c(
    "Texture",
    "Flavor",
    "Sweetness",
    "Sourness",
    "Artificial taste"
  ),
  pattern = c(
    "texture|creamy|smooth|thick",
    "flavor|taste|tasted|aftertaste",
    "sweet",
    "sour",
    "artificial"
  )
)
#and which attributes are mentioned in each comment
detected_aspects <- tidyr::crossing(
  feedback,
  aspects
) %>%
  filter(
    str_detect(
      str_to_lower(comment),
      pattern
    )
  ) %>%
  select(id, comment, aspect)

print(detected_aspects)

#apply the pretrained AI sentiment model
py_install(
  "tiktoken",
  envname = "absa_env",
  pip = TRUE
)

py_install(
  "protobuf",
  envname = "absa_env",
  pip = TRUE
)

py_run_string("
import spacy
nlp = spacy.load('en_core_web_sm')
print('spaCy loaded successfully')
")

py_run_string("
from transformers import pipeline

classifier = pipeline(
    'text-classification',
    model='yangheng/deberta-v3-base-absa-v1.1',
    device=-1
)

def classify_aspect(text, aspect):
    predictions = classifier(
        text,
        text_pair=aspect,
        top_k=None,
        truncation=True
    )

    best = max(
        predictions,
        key=lambda x: x['score']
    )

    return {
        'sentiment': best['label'],
        'confidence': best['score']
    }
")

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

#apply the model to every detected aspect
absa_results <- detected_aspects %>%
  mutate(
    prediction = map2(
      comment,
      aspect,
      ~ py_to_r(
        py$classify_aspect(.x, .y)
      )
    ),
    sentiment = map_chr(
      prediction,
      "sentiment"
    ),
    confidence = map_dbl(
      prediction,
      "confidence"
    )
  ) %>%
  select(
    id,
    comment,
    aspect,
    sentiment,
    confidence
  )

print(absa_results)

#summarize sentiment by product attribute
sentiment_summary <- absa_results %>%
  count(aspect, sentiment) %>%
  pivot_wider(
    names_from = sentiment,
    values_from = n,
    values_fill = 0
  )

print(sentiment_summary)

ggplot(
  absa_results,
  aes(
    x = aspect,
    fill = sentiment
  )
) +
  geom_bar(position = "dodge") +
  coord_flip() +
  labs(
    title = "Aspect Based Sentiment Analysis",
    subtitle = "Consumer Feedback on Yogurt",
    x = "Product Attribute",
    y = "Number of Mentions",
    fill = "Sentiment"
  ) +
  theme_minimal()

#calculate percentage of positive mentions
sentiment_percent <- absa_results %>%
  count(aspect, sentiment) %>%
  group_by(aspect) %>%
  mutate(
    percentage = 100 * n / sum(n)
  ) %>%
  ungroup()

ggplot(
  sentiment_percent,
  aes(
    x = aspect,
    y = percentage,
    fill = sentiment
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Consumer Sentiment by Product Attribute",
    x = "Product Attribute",
    y = "Percentage of Mentions",
    fill = "Sentiment"
  ) +
  theme_minimal()


#fully automated aspect discovery, 
#where an AI model identifies attributes such as texture, sweetness, packaging, and aftertaste
#without defining those categories beforehand

#start with RoBERTa ABSA Aspect Extraction and Sentiment Model
#extracts aspect terms and classifies their sentiment in one operation

use_virtualenv("absa_env", required = TRUE)

virtualenv_install(
  envname = "absa_env",
  packages = paste0(
    "https://github.com/explosion/spacy-models/releases/",
    "download/en_core_web_sm-3.8.0/",
    "en_core_web_sm-3.8.0-py3-none-any.whl"
  ),
  ignore_installed = TRUE
)
py_run_string("
import spacy
print(spacy.__version__)
")

py_install(
  c("torch", "transformers", "pandas"),
  envname = "absa_env",
  pip = TRUE
)

py_config()

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

#load the AI model
py_run_string("
from transformers import pipeline

#this model examines the text and identifies spans that represent aspects.
#for each extracted aspect, it returns a sentiment classification and model confidence score.
#there is no manually defined vocabulary of product attributes.
absa_model = pipeline(
    task='token-classification',
    model='gauneg/roberta-base-absa-ate-sentiment',
    aggregation_strategy='simple',
    device=-1
)
")

#automatically extract aspect and sentiment
py_run_string("
import pandas as pd

def extract_aspects(ids, comments):

    records = []

    for consumer_id, comment in zip(ids, comments):

        predictions = absa_model(comment)

        for item in predictions:

            records.append({
                'id': consumer_id,
                'comment': comment,
                'aspect': item['word'].strip(),
                'sentiment': item['entity_group'],
                'confidence': float(item['score'])
            })

    return pd.DataFrame(
        records,
        columns=[
            'id',
            'comment',
            'aspect',
            'sentiment',
            'confidence'
        ]
    )
")

#call the function from r
absa_results <- py_to_r(
  py$extract_aspects(
    as.list(feedback$id),
    as.list(feedback$comment)
  )
)

absa_results <- as_tibble(absa_results)

print(absa_results)

#summarize the discovered aspects
aspect_summary <- absa_results %>%
  count(aspect, sort = TRUE)

print(aspect_summary)
#summarize positive, negative, and neutral sentiment by aspect
sentiment_summary <- absa_results %>%
  mutate(
    sentiment = recode(
      tolower(sentiment),
      "pos" = "Positive",
      "neg" = "Negative",
      "neu" = "Neutral"
    )
  ) %>%
  count(aspect, sentiment)

print(sentiment_summary)

#visualize sentiment by discovered attribute
ggplot(
  sentiment_summary,
  aes(
    x = reorder(aspect, n),
    y = n,
    fill = sentiment
  )
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Automated Aspect Based Sentiment Analysis",
    subtitle = "Yogurt Consumer Feedback",
    x = "AI Discovered Attribute",
    y = "Number of Mentions",
    fill = "Sentiment"
  ) +
  theme_minimal()

#identify attributes needing attention
negative_aspects <- absa_results %>%
  filter(tolower(sentiment) == "neg") %>%
  arrange(desc(confidence)) %>%
  select(
    id,
    aspect,
    confidence,
    comment
  )

print(negative_aspects)

#it looks like the model dod not identified the sentiments correctly
#for example, "too sweet for my taste" was identified as positive

#try BART Large MNLI, a pretrained model that supports zero shot classification

py_run_string("
from transformers import pipeline

sentiment_model = pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli',
    device=-1
)
")

#create a context-aware sentiment function
py_run_string("
def classify_aspect_sentiment(comment, aspect):

    labels = [
        'satisfied with this product attribute',
        'dissatisfied with this product attribute',
        'neutral about this product attribute'
    ]

    text = (
        f'Consumer comment: {comment}\\n'
        f'Product attribute being evaluated: {aspect}'
    )

    result = sentiment_model(
        text,
        candidate_labels=labels,
        hypothesis_template='The consumer is {}.',
        multi_label=False
    )

    mapping = {
        labels[0]: 'Positive',
        labels[1]: 'Negative',
        labels[2]: 'Neutral'
    }

    return {
        'sentiment': mapping[result['labels'][0]],
        'confidence': result['scores'][0]
    }
")

#test the problematic comments 
test_cases <- tibble(
  comment = c(
    "Too sweet for my taste",
    "Too sour and not sweet enough",
    "Too sour and not sweet enough",
    "I liked the flavor but it was too thick",
    "I loved the creamy texture"
  ),
  aspect = c(
    "sweet",
    "sweet",
    "sour",
    "thick",
    "texture"
  )
)

test_results <- test_cases %>%
  mutate(
    result = map2(
      comment,
      aspect,
      ~ py_to_r(
        py$classify_aspect_sentiment(.x, .y)
      )
    ),
    sentiment = map_chr(result, "sentiment"),
    confidence = map_dbl(result, "confidence")
  ) %>%
  select(comment, aspect, sentiment, confidence)

print(test_results)

#reclassify all extracted aspects
absa_improved <- absa_results %>%
  select(id, comment, aspect) %>%
  mutate(
    result = map2(
      comment,
      aspect,
      ~ py_to_r(
        py$classify_aspect_sentiment(.x, .y)
      )
    ),
    sentiment = map_chr(result, "sentiment"),
    confidence = map_dbl(result, "confidence")
  ) %>%
  select(
    id, comment, aspect,
    sentiment, confidence
  )

print(absa_improved)

#combine the word cloud with the aspect-based sentiment analysis
#this analysis is for absa improved, but can be done for the initial absa, absa_results
absa_cloud <- absa_improved %>%
  mutate(
    aspect = str_to_lower(str_squish(aspect)),
    sentiment = str_to_title(sentiment)
  ) %>%
  filter(
    !is.na(aspect),
    aspect != "",
    sentiment %in% c(
      "Positive", "Negative", "Neutral"
    )
  )
#calculate attribute frequencies
aspect_counts <- absa_cloud %>%
  distinct(id, aspect, sentiment) %>%
  count(
    aspect,
    sentiment,
    name = "frequency"
  )

print(aspect_counts)

#create sentiment specific word clouds and generate the word clouds
#install.packages("ggwordcloud")
library(ggwordcloud)
set.seed(123)

ggplot(
  aspect_counts,
  aes(
    label = aspect,
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
  scale_size_area(max_size = 20) +
  scale_color_manual(
    values = c(
      "Positive" = "darkgreen",
      "Negative" = "red3",
      "Neutral" = "gray50"
    )
  ) +
  labs(
    title = "Consumer Sentiment by Product Attribute",
    subtitle = "Automated Aspect Based Sentiment Analysis"
  ) +
  theme_void() +
  theme(
    legend.position = "none",
    strip.text = element_text(
      size = 13,
      face = "bold"
    ),
    plot.title = element_text(
      size = 16,
      face = "bold"
    )
  )

#calculate positive and negative sentiment percentages
aspect_percent <- aspect_counts %>%
  group_by(aspect) %>%
  mutate(
    percent = 100 * frequency / sum(frequency)
  ) %>%
  ungroup()

ggplot(
  aspect_percent,
  aes(
    x = aspect,
    y = percent,
    fill = sentiment
  )
) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(
    values = c(
      "Positive" = "darkgreen",
      "Negative" = "red3",
      "Neutral" = "gray60"
    )
  ) +
  labs(
    title = "Sentiment Distribution by Attribute",
    x = "Product Attribute",
    y = "Percentage of Mentions",
    fill = "Sentiment"
  ) +
  theme_minimal()
