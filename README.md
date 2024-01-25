
# scyan

#### Description

`scyan` operator performs cell annotation and debarcoding.

#### Usage

Input Step 1|.
---|---
`row`   | represents the variables (e.g. markers, channels)
`col`   | represents the observations (e.g. event IDs) 
`y-axis`| measurement value

Input Step 2|.
---|---
`row`   | represents the variables (e.g. markers, channels). May have multiple factors to indicate hierarchy. The leftmost factor is considered the lowest levels (i.e. most detailed / leaves) of the population hierarchy tree.
`col`   | represents the observations (e.g. event IDs) 
`y-axis`| measurement value


Output|.
---|---
`PredictedPopulation`| Scores on the principal components 1..maxComp, i.e. the data projected on the principal components.


#### Reference

[Blampey et al., 2023, A biology-driven deep generative model for cell-type annotation in cytometry, Briefings in Bioinformatics, Volume 24, Issue 5, September 2023, bbad260](https://doi.org/10.1093/bib/bbad260)

[scyan](https://github.com/MICS-Lab/scyan) is distributed under [this BSD 3-Clause License](https://github.com/MICS-Lab/scyan/blob/ae0d612ecd6f59cd9a95127198d6c6f5700f1975/LICENSE).
