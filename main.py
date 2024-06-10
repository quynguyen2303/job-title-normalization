import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

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
stop_words = ["senior", "fresher", "junior", "middle", "intern", "sr", "jr", "fr", "expert", "level", "associate", "developer", "engineer", "manager"]

# Remove stop words from job title and job title table




def calculate_similarity(job_title, job_title_table, cleaned_job_title_table):
    
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


    # Tokenize the job title
    job_title_tokens = tokenizer.encode(job_title, add_special_tokens=True, truncation=True, max_length=128, padding='max_length', return_tensors='pt')

    # Tokenize the job title table
    job_title_table_tokens = tokenizer.batch_encode_plus(cleaned_job_title_table, add_special_tokens=True, truncation=True, max_length=128, padding='max_length', return_tensors='pt')

    # Get the embeddings for the job title and job title table
    with torch.no_grad():
        job_title_embeddings = model(job_title_tokens)[0]
        job_title_table_embeddings = model(job_title_table_tokens['input_ids'])[0]

    # Reshape the embeddings to have a 2-dimensional shape
    job_title_embeddings = job_title_embeddings.reshape(job_title_embeddings.shape[0], -1)
    job_title_table_embeddings = job_title_table_embeddings.reshape(job_title_table_embeddings.shape[0], -1)

    # Calculate the cosine similarity between the job title and each job title in the table
    similarity_scores = cosine_similarity(job_title_embeddings, job_title_table_embeddings)

    # Find the best similarity score
    best_score = similarity_scores.max()

    # Return the standard job title with the best similarity score
    best_index = similarity_scores.argmax()
    best_job_title = job_title_table[best_index]

    # Print all the similarity scores with the standard job titles
    for i, score in enumerate(similarity_scores[0]):
        print(f"Similarity score for '{job_title}' and '{job_title_table[i]}': {score}")

    return best_job_title, best_score
    


job_title = 'Senior Backend Engineer'
job_title = ' '.join([word for word in job_title.split() if word.lower() not in stop_words])
cleaned_job_title_table = [' '.join([word for word in title.split() if word.lower() not in stop_words]) for title in job_title_table]
best_job_title, best_score = calculate_similarity(job_title, job_title_table, cleaned_job_title_table)

print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}")