# text-summarization-S2S-Model

## Overview

A Seq2Seq model with Attention that reads a food review and generates a short summary.

## Side notes
The model is pretty bad, it was only trained on a 100 summaries so it can't make a really good summary.

Sample test:
```
Test Loss: nan

Sample predictions:

Example 1
Pred: first okay party especially rather oils lemonlime pot pot fat work approved transfer deliberately company remember loved wholesome home method
True: flavor

Example 2
Pred: first okay party especially starving doesnt sits whenever taffy year expensive hour five tea agree regularly method visits abdominal sensitive
True: stomach
```

## Prerequisites

Before you begin, ensure you have the following installed:

- pandas
- numpy
- python-dotenv
- nltk
- torch
- scikit-learn
- beautifulsoup4


## Configuration

Configuration options can be set via environment variables or a config file:

### Environment Variables

```bash
REVIEWS_DATA={Path_to_review_data}

```



### Development Setup

```bash
git clone https://github.com/Troppy2/text-summarization.git
cd [project_name]
pip install python-dotenv numpy pandas nltk torch scikit-learn beautifulsoup4
```
