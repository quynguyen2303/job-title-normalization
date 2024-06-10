def test_calculate_similarity():
    # Test case 1: Senior Backend Engineer
    job_title = 'Senior Backend Engineer'
    best_job_title, best_score = calculate_similarity(
        job_title, job_title_table)
    print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}")
    assert best_job_title == 'Back-end Engineer'

    # Test case 2: Data Scientist
    job_title = 'Data Scientist'
    best_job_title, best_score = calculate_similarity(
        job_title, job_title_table)
    print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}")
    assert best_job_title == 'Data Scientist'

    # Test case 3: UI Designer
    job_title = 'UI Designer'
    best_job_title, best_score = calculate_similarity(
        job_title, job_title_table)
    print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}")
    assert best_job_title == 'UI UX Designer'

    # Test case 4: Cloud Security Engineer
    job_title = 'Cloud Security Engineer'
    best_job_title, best_score = calculate_similarity(
        job_title, job_title_table)
    print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}")
    assert best_job_title == 'Cloud Security Engineer'

    # Test case 5: Manual Tester
    job_title = 'Manual Tester'
    best_job_title, best_score = calculate_similarity(
        job_title, job_title_table)
    print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}")
    assert best_job_title == 'Manual Tester'

    # Test case 6: Agile Coach
    job_title = 'Agile Coach'
    best_job_title, best_score = calculate_similarity(
        job_title, job_title_table)
    print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}")
    assert best_job_title == 'Agile Coach'

    # Test case 7: Software Project Manager
    job_title = 'Software Project Manager'
    best_job_title, best_score = calculate_similarity(
        job_title, job_title_table)
    print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}")
    assert best_job_title == 'Software Project Manager'

    # Test case 8: QA Engineer
    job_title = 'QA Engineer'
    best_job_title, best_score = calculate_similarity(
        job_title, job_title_table)
    print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}")
    assert best_job_title == 'Quality Assurance Engineer'

    # Test case 9: Product Owner
    job_title = 'Product Owner'
    best_job_title, best_score = calculate_similarity(
        job_title, job_title_table)
    print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}")
    assert best_job_title == 'Product Owner'

    # Test case 10: Machine Learning Engineer
    job_title = 'Machine Learning Engineer'
    best_job_title, best_score = calculate_similarity(
        job_title, job_title_table)
    print(f"The similarity score for '{job_title}' is {best_job_title} with {best_score}")
    assert best_job_title == 'Machine Learning Engineer'

if __name__ == '__main__':
    test_calculate_similarity()