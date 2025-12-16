from step_a_faces_emotions import run_faces_emotions
from step_b_activities import run_activities
from step_c_summary import generate_summary

def main():
    run_faces_emotions()
    run_activities()
    generate_summary()

if __name__ == "__main__":
    main()
