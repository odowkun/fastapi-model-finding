import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np

# Load the data
technicians_df = pd.read_csv('technicians.csv')

# Handle missing values: Fill missing certifications with an empty string
technicians_df['certifications'] = technicians_df['certifications'].fillna('')

# Encode skills and certifications as binary vectors
mlb_skills = MultiLabelBinarizer()
skills_encoded = mlb_skills.fit_transform(technicians_df['skills'].str.split(', '))

mlb_certifications = MultiLabelBinarizer()
certifications_encoded = mlb_certifications.fit_transform(technicians_df['certifications'].str.split(', '))

# Convert to DataFrames for easier manipulation
skills_df = pd.DataFrame(skills_encoded, columns=mlb_skills.classes_)
certifications_df = pd.DataFrame(certifications_encoded, columns=mlb_certifications.classes_)

# Concatenate encoded skills and certifications with the original dataframe
technicians_df_encoded = pd.concat([technicians_df, skills_df, certifications_df], axis=1)

# Drop original skills and certifications columns
technicians_df_encoded = technicians_df_encoded.drop(['skills', 'certifications'], axis=1)

# Normalize experience and ratingsreceived
scaler = StandardScaler()
technicians_df_encoded[['experience', 'ratingsreceived']] = scaler.fit_transform(technicians_df_encoded[['experience', 'ratingsreceived']])

# Define a function to calculate the composite score
def calculate_composite_score(technician, skill_needed):
    # Score for having the required skill
    skill_score = technician.get(skill_needed, 0)
    # Scores for experience and ratings
    experience_score = technician['experience']
    ratings_score = technician['ratingsreceived']
    # Score for certifications
    num_certifications = sum([technician.get(cert, 0) for cert in mlb_certifications.classes_])
    if num_certifications == 0:
        certification_score = 0
    elif num_certifications == 1:
        certification_score = 1
    else:
        certification_score = 2
    return skill_score + experience_score + ratings_score + certification_score

# Define a function to recommend a technician
def recommend_technician(skill_needed, technicians_df):
    technicians_df = technicians_df.copy()
    technicians_df['composite_score'] = technicians_df.apply(lambda x: calculate_composite_score(x, skill_needed), axis=1)
    recommended_technician_index = technicians_df.sort_values(by='composite_score', ascending=False).iloc[0].name
    recommended_technician = technicians_df.loc[recommended_technician_index]  # Retrieve details from the raw dataset
    return recommended_technician

# Split the data into training and test sets
train_df, test_df = train_test_split(technicians_df_encoded, test_size=0.2, random_state=42)

# Define the skill needed for testing
skill_needed = "Computer Maintenance"  # Example skill needed

# Perform manual cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = []

for train_index, val_index in kf.split(train_df):
    cv_train_df = train_df.iloc[train_index]
    cv_val_df = train_df.iloc[val_index]
    recommended_technician = recommend_technician(skill_needed, cv_val_df)
    cross_val_scores.append(recommended_technician['composite_score'])

print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cross_val_scores)}")

# Evaluate the model on the test set
recommended_technician = recommend_technician(skill_needed, test_df)

# Retrieve the detailed information of the recommended technician from the original dataframe
detailed_recommended_technician = technicians_df.loc[technicians_df['technicianid'] == recommended_technician['technicianid']]

print(f"Recommended Technician for skill '{skill_needed}' from the test set:")
print(detailed_recommended_technician.to_dict(orient='records')[0])