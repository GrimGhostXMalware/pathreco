def recommend_jobs(user_skills, job_data):
    scores = []
    for job in job_data:
        job_skills = job['required_skills'].split(',')
        match_score = len(set(user_skills) & set(job_skills))
        scores.append((job['job_title'], match_score))
    return sorted(scores, reverse=True)[:5]