# Dataset

Datasets after preprocessing can be downloaded online: [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/ad8582198c0f4df39d93/)

* `Preprocess.ipynb` generates datasets from original data files.
* `Data Analysis.ipynb` analyzes data to reveal temporal cross-effects in knowledge tracing (conditional mutual information analyses in the empirical study).
* `Param Analysis.ipynb` loads the dumped parameters and calculates prerequisite scores for each skill.

If you want to run the codes on a new dataset, there should be a file named `interaction.csv` in the dataset directory with the format: `user_id \t skill_id \t problem_id \t timestamp`. Each line is an interaction record and make sure the records are sorted by the timestamp in an ascending order.

Besides, the generated datasets for ASSISTments also contain additional features (student, school, et.al.), which are not incorporated in HawkesKT but can be used for further researches.

