import json

def generate_qna(data):
    qna_pairs = []
    cv_data = data['cv_data']

    for exp in cv_data['experience']:
        role = exp['role']
        company = exp['company']
        description = "\n".join(exp['description'])  # Combine description lines
        tags = ", ".join(exp['tags']) #Combine tags
        
        qna_pairs.append({"question": f"What were Andrew Cabrera's responsibilities at {company}?", "answer": description})
        qna_pairs.append({"question": f"What technologies did Andrew use while working as a {role}?", "answer": tags})
        qna_pairs.append({"question": f"What was Andrew Cabrera's role at {company}?", "answer": role})
        qna_pairs.append({"question": f"Where did Andrew Cabrera work as a {role}?", "answer": company})


    for skill_cat in cv_data['skills']:
      category = skill_cat["category"]
      technologies = ", ".join(skill_cat["technologies"])
      qna_pairs.append({"question": f"What {category} technologies does Andrew Cabrera know?", "answer": technologies})

      for edu in cv_data['education']:
        degree = edu['degree']
        institution = edu['institution']
        field = edu['field']
        dates = edu['dates']
        
        qna_pairs.append({"question": f"What degree did Andrew Cabrera earn from {institution}?", "answer": degree})
        qna_pairs.append({"question": f"What was Andrew Cabrera's major at {institution}?", "answer": field})
        qna_pairs.append({"question": f"When did Andrew Cabrera attend {institution}?", "answer": dates})

    for personal in cv_data['personal_info']:
        name = personal['name']
        location = personal['contact']['location']
        email = personal['contact']['email']
        summary = personal['summary']
        languages = personal['languages']
        
        qna_pairs.append({"question": f"Where is {name} based?", "answer": location})
        qna_pairs.append({"question": f"What is {name}'s summary?", "answer": summary})
        qna_pairs.append({"question": f"How many languages does {name} speak?", "answer": languages})
        qna_pairs.append({"question": f"What is {name}'s email address?", "answer": email})
    
    qna_pairs.append({"question": f"How was this website built?", "answer": "The website was built in Rust using Axum for the backend, Leptos for the frontend, Bevy for the visualizations, and wasm-bindgen for custom Babylon.js bindings and audio processor used to play Amiga modules."})

    return qna_pairs

if __name__ == "__main__":
    with open("dataset/cv_dataset.json", "r") as f:
        data = json.load(f)

    qna_data = generate_qna(data)

    # Save Q&A data to a new JSON file
    with open("qa_dataset.json", "w") as outfile:
        json.dump(qna_data, outfile, indent=4)

    print("Q&A dataset generated and saved to qa_dataset.json")
