# Week 1 Day 2 - Transformers Pipeline

## What is pipeline?

pipeline combines:

1. preprocessing
2. model inference
3. postprocessing

## Tasks practiced

- sentiment-analysis
- text-generation
- ner

## Important concepts

### Model ID

Example:

distilbert/distilbert-base-uncased-finetuned-sst-2-english

A Model ID points to a model repository on Hugging Face Hub.

### Pipeline structure

Input
→ Tokenizer
→ Model
→ Output
→ Postprocessing

## Questions

- What is a tokenizer?
- What happens behind pipeline?
- Why do different tasks need different models?