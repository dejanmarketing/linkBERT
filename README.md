---
language: en
license: bigscience-bloom-rail-1.0
datasets:
- bookcorpus
- wikipedia
pipeline_tag: token-classification
---

# LinkBERT: Fine-tuned BERT for Natural Link Prediction

[LinkBERT](https://huggingface.co/dejanseo/LinkBERT) is an advanced fine-tuned version of the [bert-large-cased](https://huggingface.co/google-bert/bert-large-cased) model developed by [Dejan Marketing](https://dejanmarketing.com/). The model is designed to predict natural link placement within web content. This binary classification model excels in identifying distinct token ranges that web authors are likely to choose as anchor text for links. By analyzing never-before-seen texts, LinkBERT can predict areas within the content where links might naturally occur, effectively simulating web author behavior in link creation.

# Online Demo

Online demo of this model is available at https://dejan.ai/linkbert/

## Applications of LinkBERT

LinkBERT's applications are vast and diverse, tailored to enhance both the efficiency and quality of web content creation and analysis:

- **Anchor Text Suggestion:** Acts as a mechanism during internal link optimization, suggesting potential anchor texts to web authors.
- **Evaluation of Existing Links:** Assesses the naturalness of link placements within existing content, aiding in the refinement of web pages.
- **Link Placement Guide:** Offers guidance to link builders by suggesting optimal placement for links within content.
- **Anchor Text Idea Generator:** Provides creative anchor text suggestions to enrich content and improve SEO strategies.
- **Spam and Inorganic SEO Detection:** Helps identify unnatural link patterns, contributing to the detection of spam and inorganic SEO tactics.

## Training and Performance

LinkBERT was fine-tuned on a dataset of organic web content and editorial links. The training involved preprocessing web content, annotating links with temporary markup for clear distinction, and employing a specialized tokenization process to prepare the data for model training.

### Training Highlights:

- **Dataset:** Custom organic web content with editorial links.
- **Preprocessing:** Links annotated with `[START_LINK]` and `[END_LINK]` markup.
- **Tokenization:** Utilized input_ids, token_type_ids, attention_mask, and labels for model training, with a unique labeling system to differentiate between link/anchor text and plain text.

### Technical Specifications:

- **Batch Size:** 10, with class weights adjusted to address class imbalance between link and plain text.
- **Optimizer:** AdamW with a learning rate of 5e-5.
- **Epochs:** 5, incorporating gradient accumulation and warmup steps to optimize training outcomes.
- **Hardware:** 1 x RTX4090 24GB VRAM
- **Duration:** 32 hours

## Utilization and Integration

LinkBERT is positioned as a powerful tool for content creators, SEO specialists, and webmasters, offering unparalleled support in optimizing web content for both user engagement and search engine recognition. Its predictive capabilities not only streamline the content creation process but also offer insights into the natural integration of links, enhancing the overall quality and relevance of web content.

## Accessibility

LinkBERT leverages the robust architecture of bert-large-cased, enhancing it with capabilities specifically tailored for web content analysis. This model represents a significant advancement in the understanding and generation of web content, providing a nuanced approach to natural link prediction and anchor text suggestion.

---

# BERT large model (cased)

Pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in
[this paper](https://arxiv.org/abs/1810.04805) and first released in
[this repository](https://github.com/google-research/bert). This model is cased: it makes a difference
between english and English.

Disclaimer: The team releasing BERT did not write a model card for this model so this model card has been written by
the Hugging Face team.

## Model description

BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it
was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it
was pretrained with two objectives:

- Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run
  the entire masked sentence through the model and has to predict the masked words. This is different from traditional
  recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like
  GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the
  sentence.
- Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes
  they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to
  predict if the two sentences were following each other or not.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard
classifier using the features produced by the BERT model as inputs.

This model has the following configuration:

- 24-layer
- 1024 hidden dimension
- 16 attention heads
- 336M parameters.
