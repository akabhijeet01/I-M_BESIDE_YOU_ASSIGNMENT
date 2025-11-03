# email_assistant2.py
from generate_email import generate_personalized_email, save_email_to_file
from evaluate_generation2 import evaluate_model

def interactive_email_session():
    """Interactive CLI for email generation."""
    print("\n Welcome to the Email Assistant!\n")

    while True:
        prof_name = input("Professor's Name: ")
        subject = input("Email Subject: ")
        student_bio = input("Your Short Bio: ")
        goal = input("Your Goal (internship, project, etc.): ")
        tone = input("Tone (formal/polite/friendly) [default=formal]: ") or "formal"

        fields = {
            "prof_name": prof_name,
            "subject": subject,
            "student_bio": student_bio,
            "goal": goal
        }

        print("\n🪄 Generating email...\n")
        email_text = generate_personalized_email(fields, tone)
        print("\n----- GENERATED EMAIL -----\n")
        print(email_text)
        save_email_to_file(email_text, fields)

        again = input("\nGenerate another email? (y/n): ").lower()
        if again != "y":
            break

def main():
    print("\n=== Email Assistant Main Menu ===")
    print("1️  Interactive Email Generator")
    print("2️  Evaluate Model on valid.jsonl")
    choice = input("\nEnter choice (1 or 2): ")

    if choice == "1":
        interactive_email_session()
    elif choice == "2":
        evaluate_model()
    else:
        print(" Invalid choice.")

if __name__ == "__main__":
    main()
