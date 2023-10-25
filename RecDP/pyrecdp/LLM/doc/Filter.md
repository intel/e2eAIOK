# RecDP LLM - Filters API

| Name                    | Distribution                                                                                                                                                                                             |
|:------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AlphanumericFilter      | Keeps samples with alphanumeric ratio within the specified range                                                                                                                                         |
| AverageLineLengthFilter | Keeps samples with average line length within the specified range                                                                                                                                        |
| BadwordsFilter          | Keeps samples without bad words. The bad words list comes from [List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) |
| LengthFilter            | Keeps samples with total text length within the specified range                                                                                                                                          |
| MaximumLineLengthFilter | Keeps samples with maximum line length within the specified range                                                                                                                                        |
| PerplexityFilter        | Keeps samples with perplexity score below the specified threshold                                                                                                                                        |
| ProfanityFilter         | Keeps sample without profanity language. Mainly using [alt-profanity-check](https://pypi.org/project/alt-profanity-check/) library                                                                       |
| SpecialCharactersFilter | Keeps samples with special-char ratio within the specified range                                                                                                                                         |
| TokenNumFilter          | Keeps samples with token count within the specified range                                                                                                                                                |
| URLFilter               | Keeps samples according to URLs based on [blacklist](https://dsi.ut-capitole.fr/blacklists/)                                                                                                             |
| WordNumFilter           | Keeps samples with word count within the specified range                                                                                                                                                 |
| WordRepetitionFilter    | Keeps samples with word-level n-gram repetition ratio within the specified range                                                                                                                         |

## AlphanumericFilter

| Parameters | Distribution                                                                                                                |
|:-----------|:----------------------------------------------------------------------------------------------------------------------------|
| min_ratio  | The min filter ratio, samples will be filtered if their alphabet/numeric ratio is below this parameter. Default: 0.25       |
| max_ratio  | The max filter ratio, samples will be filtered if their alphabet/numeric ratio exceeds this parameter. Default: sys.maxsize |


## AverageLineLengthFilter

| Parameters | Distribution                                                                                                               |
|:-----------|:---------------------------------------------------------------------------------------------------------------------------|
| min_len    | The min filter length, samples will be filtered if their average line length is below this parameter. Default: 10          |
| max_len    | The max filter length, samples will be filtered if their average line length exceeds this parameter. Default: sys.maxsize  |

## BadwordsFilter

| Parameters     | Distribution                                                                                                                                                                                                           |
|:---------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| language       | Sample in which language. Default: en. Referring [List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) to get supported language   |


## LengthFilter

| Parameters | Distribution                                                                                                                        |
|:-----------|:------------------------------------------------------------------------------------------------------------------------------------|
| min_len    | The min text length in the filtering. samples will be filtered if their text length is below this parameter. Default: 100           |
| max_len    | The max text length in the filtering. samples will be filtered if their text length exceeds this parameter. Default: -1(unlimited)  |

## MaximumLineLengthFilter

| Parameters   | Distribution                                                                                                                 |
|:-------------|:-----------------------------------------------------------------------------------------------------------------------------|
| min_len      | The min filter length, samples will be filtered if their maximum line length is below this parameter. Default: 10            |
| max_len      | The max filter length, samples will be filtered if their maximum line length exceeds this parameter. Default: sys.maxsize    |

## PerplexityFilter

| Parameters | Distribution                                                                                                  |
|:-----------|:--------------------------------------------------------------------------------------------------------------|
| language   | Sample in which language. Default: en. (en, zh)                                                               | 
| max_ppl    | The max filter perplexity, samples will be filtered if their perplexity exceeds this parameter. Default: 1500 |

## ProfanityFilter

### Notes:
> We also provide a similar op: TextToxity. Compared to TextToxity, ProfanityFilter main ability is to check for profanity or offensive language in text and filter out them.
> ProfanityFilter depends on the library profanity-check. This library use a linear SVM model trained on 200k human-labeled samples of clean and profane text strings. Its model is simple but surprisingly effective, meaning profanity-check is both robust and extremely performant.

| Parameters   | Distribution                                                                                                                    |
|:-------------|:--------------------------------------------------------------------------------------------------------------------------------|
| threshold    | The max profanity threshold, samples will be filtered if their profanity score exceeds this parameter. Default: 0.0 (Float 0-1) |

## SpecialCharactersFilter

| Parameters  | Distribution                                                                                                       |
|:------------|:-------------------------------------------------------------------------------------------------------------------|
| min_ratio   | The min filter ratio, samples will be filtered if their special-char ratio is below this parameter. Default: 0.0   |
| max_ratio   | The max filter ratio, samples will be filtered if their special-char ratio exceeds this parameter. Default: 0.25   |

## TokenNumFilter

| Parameters  | Distribution                                                                                                               |
|:------------|:---------------------------------------------------------------------------------------------------------------------------|
| min_num     | The min filter token number, samples will be filtered if their token number is below this parameter. Default: 10           |
| max_num     | The max filter token number, samples will be filtered if their token number exceeds this parameter. Default: sys.maxsize   |
| model_key   | The tokenizer name of Hugging Face tokenizers. Default: _EleutherAI/pythia-6.9b-deduped_                                   |


## WordNumFilter

| Parameters  | Distribution                                                                                                           |
|:------------|:-----------------------------------------------------------------------------------------------------------------------|
| min_num     | The min filter word number, samples will be filtered if their word number is below this parameter. Default: 10         |
| max_num     | The max filter word number, samples will be filtered if their word number exceeds this parameter. Default: sys.maxsize |
| language    | Sample in which language. Default: en. (en, zh)                                                                        |


## WordRepetitionFilter

| Parameters      | Distribution                                                                                                                          |
|:----------------|:--------------------------------------------------------------------------------------------------------------------------------------|
| language        | Sample in which language. Default: en.                                                                                                |
| rep_len         | Repetition length for word-level n-gram. Default: 10                                                                                  |
| min_ratio       | The min filter ratio, samples will be filtered if their word-level n-gram repetition ratio is below this parameter. Default: 0.0      |
| max_ratio       | The max filter ratio, samples will be filtered if their word-level n-gram repetition ratio exceeds this parameter. Default: 0.5       |