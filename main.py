from agents.orchestrator import Orchestrator

def main():
    # Initialize the orchestrator
    orchestrator = Orchestrator()
    
    # Define the user query
    # user_query = """
    # I need to conduct a comprehensive CV analysis and comparison for a hiring process. Here are the specific requirements:

    # TASK OVERVIEW:
    # Compare all candidate CVs against a job brief to identify the most suitable candidates.

    # INPUT SOURCES:
    # - CV Files: All .txt files located in the directory "./cvs" 
    # - Job Brief: The file "brief2.txt" containing the job description and requirements

    # PROCESSING REQUIREMENTS:
    # 1. Extract content from brief2.txt as the job brief content
    # 2. Analyze criteria into "must have" and "nice to have" requirements
    # 3. Analyze each CV from the ./cvs directory against these criteria
    # 4. Score candidates based on how well they match the requirements
    # 5. Rank candidates from highest to lowest scores

    # OUTPUT EXPECTATIONS:
    # - A ranked list of top candidates
    # - Detailed scoring breakdown for each candidate
    # - Clear indication of which criteria each candidate meets or lacks
    # - Comparison table showing all candidates' performance
    # - Recommendation for the best candidates to consider for interviews

    # IMPORTANT CONSTRAINTS:
    # - Use the exact file paths provided: "./cvs" for CV directory and "brief2.txt" for job brief
    # - Do not modify or assume different file locations
    # - Include all .txt files found in the CVs directory
    # - Provide actionable insights for hiring decisions

    # Please execute this CV comparison workflow and provide comprehensive results.
    # """

    user_query = "hello, I want to do something"
    
    
    # Run the orchestrator
    final_result = orchestrator.run()

    print("=" * 50)
    print("🔄 Orchestrator Workflow Result")
    print(final_result)
    print("=" * 50)

if __name__ == "__main__":
    main()
