"""
Constant values that are used in classes and methods
"""

# Input and output directories and files
asset_path = '../../assets/Clinical/'
numeric_file = f'{asset_path}New_Metrics.csv'
label_file = f'{asset_path}main_labels.csv'
generated_path = '../../generated/Clinical/'

# Set the sampling type
sampling_type = 'smote'
# sampling_type = 'rus'

# Target label column names
label_name = 'participant_category'
# label_name = 'group_categorization'
# label_name = 'suicide'

# Defined target labels from input data
# Examples below correspond with the respective label column names
label_categories = ['1.Control', '2.Patient']
# label_categories = ['1.Control', '3.DID', '4.PTSD', '5.PTSDDS']
# label_categories = ['1.No', '2.Yes']

# Set the number of features to be selected in RFE
# Default is 'all' features
n_features = 'all'
# n_features = 7

# Set the number of splits for K-fold CV
splits = 10     # K-fold CV
test_size = 5   # 10 or 5 depending on sample size

# Universal random state
random_state = 42
