# BirdCLEF 2026 - Competition Overview

## Competition Details

**Name:** BirdCLEF 2026 - Pantanal Wildlife Audio Classification  
**Platform:** Kaggle  
**Link:** https://www.kaggle.com/competitions/birdclef-2026

## Timeline

- **Start Date:** March 11, 2026
- **Entry Deadline:** May 27, 2026
- **Team Merger Deadline:** May 27, 2026
- **Final Submission Deadline:** June 3, 2026

## Objective

Develop machine learning frameworks capable of identifying understudied species within continuous audio data from Brazil's Pantanal wetlands. The goal is to support biodiversity monitoring in one of the world's most diverse and threatened ecosystems.

## Challenge

Build models that automatically identify wildlife species (birds, amphibians, mammals, reptiles, insects) from their vocalizations in audio recordings collected across the Pantanal wetlands.

## Evaluation Metric

**Macro-averaged ROC-AUC** that skips classes with no true positive labels.

## Dataset

### Training Data

**train_audio/** - Short recordings of individual species sounds from xeno-canto.org and iNaturalist
- Resampled to 32 kHz
- OGG format
- Filenames: `[collection][file_id_in_collection].ogg`

**train_soundscapes/** - Additional audio from recording locations similar to test set
- 1-minute recordings
- Some labeled by expert annotators
- Labels provided in `train_soundscapes_labels.csv`

**train.csv** - Metadata for training data
- `primary_label`: Species code (eBird code for birds, iNaturalist taxon ID for non-birds)
- `secondary_labels`: Additional species in the recording
- `latitude` & `longitude`: Recording location
- `author`: Recordist
- `filename`: Audio filename
- `rating`: Quality rating (1-5)
- `collection`: XC or iNat

**train_soundscapes_labels.csv** - Ground truth for labeled soundscapes
- `filename`: Soundscape file
- `start` & `end`: 5-second segment timestamps
- `primary_label`: Semicolon-separated species codes

### Test Data

**test_soundscapes/** - ~600 recordings for scoring
- 1-minute long, OGG format, 32 kHz
- Filename format: `BC2026_Test_<file ID>_<site>_<date>_<time in UTC>.ogg`
- Hidden test set (populated when notebook is submitted)

### Metadata

**taxonomy.csv** - Species information
- 234 species/classes
- iNaturalist taxon ID and class name
- Includes insect sonotypes (e.g., `47158son16`)

**sample_submission.csv** - Submission format
- `row_id`: `[soundscape_filename]_[end_time]`
- 234 species columns with probability predictions

## Submission Format

For each `row_id`, predict the probability (0-1) that each of 234 species was present in the 5-second audio segment.

## Code Requirements

- **Code Competition:** Submissions via Notebooks only
- **CPU Notebook:** ≤ 90 minutes runtime
- **GPU Notebook:** Disabled (1 minute runtime only)
- **Internet:** Disabled
- **External Data:** Freely & publicly available data allowed, including pre-trained models
- **Submission File:** Must be named `submission.csv`

## Key Insights

1. **Domain Match:** Train soundscapes are from the same recording locations as test data
2. **Multi-label:** Multiple species can be present in a single segment
3. **Class Imbalance:** Some species are rare in the training data
4. **Audio Quality:** Varies from high-quality recordings to noisy field recordings
5. **Temporal Patterns:** Species may have temporal patterns across segments

## Resources

- **Competition Page:** https://www.kaggle.com/competitions/birdclef-2026
- **Discussion Forum:** https://www.kaggle.com/competitions/birdclef-2026/discussion
- **Data Page:** https://www.kaggle.com/competitions/birdclef-2026/data
- **eBird Species:** https://ebird.org/species/[code]
- **iNaturalist Taxa:** https://www.inaturalist.org/taxa/[id]
