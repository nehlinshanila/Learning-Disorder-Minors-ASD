<h2 align="center"> Learning Disorder Prediction in Pediatrics with ASD :hospital: </h2>


This is a group research project of our CSE445 (Machine Learning Project) course.

## About
Learning disorders are common among pediatric individuals with Autism Spectrum Disorder (ASD). Although autism itself is not a learning disability, it can significantly affect a child’s ability to process and retain information. Early diagnosis of pediatric individuals who may develop a learning disorder can enable effective treatment. This project aimed to develop mechanisms to detect learning disorders in pediatric individuals aged 1 to 18 years. Eight machine learning algorithms were tested to curate an accurate model to predict whether a pediatric individual is susceptible to developing a learning disorder. Random Forest outperformed all the classifiers. The model was calibrated with a tool that interprets and retraces the underlying factors contributing to the results. Experimental results revealed high reliability in detecting learning disorders in pediatric individuals.

## Index Terms
- Autism Spectrum Disorder (ASD)
- Learning Disorder
- Pediatric
- Machine Learning
- Explainable AI

## Repository Contents (useful)
- **Learning_Disorder_cse445.ipynb**: Jupyter Notebook containing the code for the machine learning models and analysis.
- **data_raw.csv**: Raw dataset used for the research.


## Data Process & Acquisition
(i) Acquisition of the dataset,
(ii) Pre-processing of the data,
(iii) Application of machine learning algorithms &
(iv) Tuning of hyper-parameters on the training set.
Dataset utilized for the research was acquired from an autism research study conducted by the University of Arkansas, containing 1985 records of pediatrics with or without ASD in the age range of 1-18 years old, mostly from Europe, Asia, and the Middle East. More information can be found here --> https://www.kaggle.com/datasets/uppulurimadhuri/dataset The dataset was partitioned into training (80%) and test (20%) sets. Machine learning techniques were utilized to build models using the training data, and their effectiveness was assessed using the test set. Finally, Explainable AI methodologies were utilized on the most optimal model to gain a deeper understanding of the causes behind the projected results.

## Pre-processing
Data pre-processing involved cleaning the dataset by removing null entries, resulting in 1937 rows and 28 features. Label encoding was used to convert categorical values to numerical representations.

## Algorithms
Machine learning algorithms applied:
- K-Nearest Neighbor (KNN)
- Random Forest
- Decision Tree
- Naïve Bayes (Gaussian)
- Logistic Regression
- ZeroR Classifiers

Random Forest emerged as the top-performing algorithm, consistently surpassing its counterparts in terms of accuracy and diagnostic abilities.

## Hyper-parameter Tuning
Two hyperparameter tuning techniques – randomized search with cross-validation and grid search with cross-validation – were used to find the best combination of hyperparameters.

## Experimental Results
The performance of various machine learning algorithms was measured using accuracy, precision, recall, and F1 score. Random Forest achieved an accuracy of 99.22%, precision of 100%, recall of 99%, and F1 score of 99%. Explainable AI techniques (LIME) provided insights into the model's predictions, highlighting factors like speech delay, social/behavioral issues, and anxiety disorder as influential in predicting learning disorders.

## Getting Started

### Prerequisites
Make sure you have the following installed on your system:
- Python 3.6 or higher
- Jupyter Notebook
- Required Python packages (listed below)

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/nehlinshanila/Learning-Disorder-Minors-ASD.git
   cd Learning-Disorder-Minors-ASD
   ```
   
2. Install the required Python packages:
   ```sh
   pip install pandas numpy matplotlib seaborn scikit-learn lime ydata-profiling
   pip install notebook
   ```

## Usage
1. Open Jupyter Notebook:
2. Navigate to and open Learning_Disorder_cse445.ipynb in Jupyter Notebook.
3. Run the notebook cells sequentially to execute the code, train models, and view results.

## Results
Table I: Accuracy for Different Algorithms on Dataset
| Algorithms            | Accuracy | Precision | Recall | F1 Score | Train time(ms) |
|-----------------------|----------|-----------|--------|----------|----------------|
| K-NN (n=5)            | 98.38%   | 100.00%   | 98.00% | 99.00%   | 0.0            |
| K-NN (n=15)           | 97.67%   | 100.00%   | 97.00% | 98.00%   | 93.8           |
| Decision-Tree (gini)  | 99.22%   | 98.00%    | 99.00% | 98.00%   | 15.6           |
| Decision-Tree (entropy)| 99.22%  | 98.00%    | 99.00% | 99.00%   | 0.0            |
| Random-Forest         | 99.22%   | 100.00%   | 99.00% | 99.00%   | 156.3          |
| Naïve-Bayes (Gaussian)| 99.03%   | 100.00%   | 99.00% | 100.00%  | 93.8           |
| ZeroR                 | 55.19%   | 54.00%    | 100.00%| 70.00%   | 0.0            |
| Logistic-Regression   | 99.03%   | 100.00%   | 99.01% | 99.00%   | 15.6           |

Table II: Accuracy for Different Algorithms on Dataset After Using Various Hyperparameter Optimizers
| Algorithms              | RandomizedSearch CV(%) | GridSearch CV(%) |
|-------------------------|------------------------|------------------|
| Random Forest (gini)    | 99.22%                 | 99.22%           |
| Random Forest (entropy) | 99.22%                 | 99.22%           |


## Conclusion and Future Work
This study presents various machine learning methods for predicting learning disorders among pediatric individuals with ASD traits, achieving remarkable accuracies. Future work includes obtaining a more extensive and diverse dataset, potentially focusing on pediatric individuals with ASD traits in Bangladesh, to further enhance prediction accuracy.


You can read more about our research in our paper published on IEEE Xplore: [Read the paper](https://ieeexplore.ieee.org/document/10499515).

### Acknowledgments
We would like to extend our gratitude to all contributors who made this research possible:
- [Shakirul Islam Leeon](https://github.com/shakirul360)
- [Fahrin Hossain Sunaira](https://github.com/Sunaira1101)
- [S.A.M. Zahin Abdal](https://github.com/ZahinSam1)
- Dr. Sifat Momen

## Cite This Paper
If you use this code and model for your research, please consider citing:
   ```sh
@INPROCEEDINGS{10499515,
  author={Leeon, Shakirul Islam and Sunaira, Fahrin Hossain and Nehlin, Shanila and Abdal, S.A.M. Zahin and Momen, Sifat},
  booktitle={2024 International Conference on Advances in Computing, Communication, Electrical, and Smart Systems (iCACCESS)}, 
  title={A Machine Learning Approach for Early Detection of Learning Disorders in Pediatrics}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Autism; Pediatrics; Machine learning algorithms; Computational modeling; Predictive models; Aging; Reliability; Autism Spectrum Disorder; Learning Disorder; Pediatric},
  doi={10.1109/iCACCESS61735.2024.10499515}
}
   ```

Thank you!











   
