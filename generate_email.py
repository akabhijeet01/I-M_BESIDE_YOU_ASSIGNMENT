# generate_email.py
import os
from datetime import datetime
from llm_adapter import generate


def build_instruction_prompt(fields: dict, tone="formal"):
    """
    Convert user input fields into a structured instruction prompt.
    """
    body = "\n".join([f"{k}: {v}" for k, v in fields.items()])
    return f"### Instruction:\nWrite a {tone} outreach email using:\n{body}\n\n### Response:\n"


def extract_response(text: str):
    """
    Clean raw LLM output to only extract response.
    """
    if "### Response:" in text:
        return text.split("### Response:")[-1].strip()
    return text.strip()


def generate_personalized_email(fields: dict, tone="formal"):
    """
    Generate the final personalized email from user-provided info.
    """
    prompt = build_instruction_prompt(fields, tone)
    raw_output = generate(prompt)
    return extract_response(raw_output)


def save_email_to_file(email_text: str, fields: dict, output_dir="generated_emails"):
    """
    Save generated email with timestamp and metadata.
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subject = fields.get("subject", "email").replace(" ", "").replace("/", "")
    filename = f"{timestamp}_{subject}.txt"
    filepath = os.path.join(output_dir, filename)

    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("----- EMAIL DETAILS -----\n")
        for k, v in fields.items():
            f.write(f"{k}: {v}\n")
        f.write("\n----- GENERATED EMAIL -----\n")
        f.write(email_text)

    print(f"\n Email saved to: {filepath}")


if __name__== "__main__":
    test_fields = {
        "prof_name": "Prof. Rohit Sharma",
        "subject": "Research Collaboration in Quantum Optics",
        "student_bio": "Abhijeet Kumar, 3rd-year Physics student at IIT Roorkee",
        "goal": "to work under your guidance this summer"
    }

    email = generate_personalized_email(test_fields)
    print("----- Generated Email -----\n")
    print(email)

    # Save the generated email automatically
    save_email_to_file(email, test_fields)