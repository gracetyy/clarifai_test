import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()
CLARIFAI_PAT = os.getenv('CLARIFAI_PAT')
clarifai_llm = LLM(
    model="openai/deepseek-ai/deepseek-chat/models/DeepSeek-R1-Distill-Qwen-7B",   
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
    api_key=os.environ["CLARIFAI_PAT"]  # Ensure this environment variable is set
)

# Define the Researcher Agent
researcher = Agent(
    role="Lead Intelligence Strategist",
    goal="Deliver creative, actionable intelligence for high-stakes business and policy decisions.",
    backstory="""You are a lead research analyst at Atlas Insights, a boutique firm renowned for solving complex, high-stakes problems for Fortune 500 companies and global NGOs. Your trademark is blending investigative journalism techniques with data-driven analysis, uncovering hidden patterns and actionable intelligence. Clients trust you for your relentless pursuit of the truth, creative research strategies, and ability to translate findings into clear, impactful recommendations tailored for decision-makers.""",    
    verbose=True, # Set to False to disable verbose output 
    allow_delegation=False,
    llm=clarifai_llm
)

def create_research_task(topic):
    return Task(
        description=f"""Prepare a confidential intelligence briefing on '{topic}'. Identify emerging risks, opportunities, influential players, and strategic recommendations. Use investigative and data-driven methods. Avoid generic statements; focus on unique, actionable insights.""",
        expected_output="A concise, executive-level intelligence briefing with bullet points, strategic recommendations, and cited sources where possible.",
        agent=researcher
    )

def run_research(topic):
    task = create_research_task(topic)

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff() # Starts the execution of the crew

if __name__ == "__main__":
    topic = input("Enter the topic to research: ").strip()
    if topic:
        print(f"\n Researching '{topic}'...\n")
        result = run_research(topic)
        print("\n Research Completed:\n")
        print(result)
    else:
        print("No topic entered.")