import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# Define standard job title table
job_title_table = [
    'Game Designer',
    'Graphic Designer',
    'Mobile Designer',
    'UI UX Designer',
    'Visual Designer',
    'Web Designer',
    'Embedded Software Engineer',
    'Mobile Engineer',
    'Full-stack Engineer',
    'Front-end Engineer',
    'Back-end Engineer',
    'Machine Learning Engineer',
    'AI Engineer',
    'Cloud Engineer',
    'Cloud Security Engineer',
    'Computer Vision Engineer',
    'Cybersecurity Engineer',
    'Data Analyst',
    'Database Administrator',
    'Data Engineer',
    'Data Scientist',
    'DevOps Engineer',
    'Site Reliability Engineer',
    'Software Engineer',
    'Product Manager',
    'Business Analyst',
    'Product Owner',
    'Agile Coach',
    'Scrum Master',
    'Project Manager',
    'IT Project Manager',
    'Software Project Manager',
    'Automation Test Engineer',
    'Manual Test Engineer',
    'Embedded Software Tester',
    'Manual Tester',
    'Automation Tester',
    'Quality Assurance Engineer',
    'QA QC Engineer'
]

# Define stop words
stop_words = ["senior", "fresher", "junior", "middle", "intern", "sr", "jr",
              "fr", "expert", "level", "associate"]#, "developer", "engineer", "manager"]

model_name = 'all-MiniLM-L6-v2'

# Create a function to clean job titles
def clean_job_title(job_title, stop_words):
    return ' '.join([word for word in job_title.split() if word.lower() not in stop_words])

# Create a function to clean job title table
def clean_job_title_table(job_title_table, stop_words):
    return [' '.join([word for word in title.split() if word.lower() not in stop_words]) for title in job_title_table]

def calculate_similarity(job_title, job_title_table):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    cleaned_job_title_table = clean_job_title_table(job_title_table, stop_words)
    cleaned_job_title = clean_job_title(job_title, stop_words)

    # Tokenize the job title
    job_title_tokens = tokenizer.encode(cleaned_job_title, add_special_tokens=True,
                                        truncation=True, max_length=128, padding='max_length', return_tensors='pt')

    # Tokenize the job title table
    job_title_table_tokens = tokenizer.batch_encode_plus(
        cleaned_job_title_table, add_special_tokens=True, truncation=True, max_length=128, padding='max_length', return_tensors='pt')

    # Get the embeddings for the job title and job title table
    with torch.no_grad():
        job_title_embeddings = model(job_title_tokens)[0]
        job_title_table_embeddings = model(job_title_table_tokens['input_ids'])[0]

    # Reshape the embeddings to have a 2-dimensional shape
    job_title_embeddings = job_title_embeddings.reshape(job_title_embeddings.shape[0], -1)
    job_title_table_embeddings = job_title_table_embeddings.reshape(job_title_table_embeddings.shape[0], -1)

    # Calculate the cosine similarity between the job title and each job title in the table
    similarity_scores = cosine_similarity(job_title_embeddings, job_title_table_embeddings)

    # Store all the scores in a dictionary with the job title as the key
    similarity_scores_dict = {job_title: score for job_title, score in zip(job_title_table, similarity_scores[0])}

    # Find the best similarity score
    best_score = similarity_scores.max()

    # Return the standard job title with the best similarity score
    best_index = similarity_scores.argmax()
    best_cleaned_job_title = cleaned_job_title_table[best_index]
    best_job_title = job_title_table[best_index]

    return cleaned_job_title, best_cleaned_job_title, best_job_title, best_score, similarity_scores_dict

def test_calculate_similarity():
    test_cases = [
        'Senior Backend Engineer',
        "Senior Frontend Engineer",
        "Java Developer",
        "Python Engineer",
        'Data Scientist',
        'UI Designer',
        'Cloud Security Engineer',
        'Manual Tester',
        'Agile Coach',
        'Software Project Manager',
        'QA Engineer',
        'Product Owner',
        'Machine Learning Engineer'
    ]

    for job_title in test_cases:
        cleaned_job_title, best_cleaned_job_title, best_job_title, best_score, similarity_scores_dict = calculate_similarity(job_title, job_title_table)
        print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}, clean job title is {cleaned_job_title} and best cleaned job title is {best_cleaned_job_title}")
        # Add current date to the result
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Write out a csv file with the results
        with open('result.csv', 'a') as f:
            f.write(f"{model_name},{job_title},{best_job_title},{best_score},{cleaned_job_title},{best_cleaned_job_title},{current_date}\n")
        # store the similarity scores in a csv file
        with open('similarity_scores.csv', 'a') as f:
            for job_title, score in similarity_scores_dict.items():
                f.write(f"{model_name},{cleaned_job_title},{job_title},{score},{current_date}\n")

if __name__ == '__main__':
    test_calculate_similarity()
