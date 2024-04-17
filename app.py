import streamlit as st
from openai import OpenAI
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Assuming you have already set up the API key for OpenAI in your environment variables
api_key = api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
        st.error("OpenAI API key is not set. Please set it in your environment variables.")
        
client = OpenAI(api_key=api_key)

#data decorator
@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename)
    return df

# Graphing Cache
@st.cache_data
def plot_distributions(df, selected_name, columns):
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # Adjust layout size based on your need
        axes = axes.flatten()
        
        for i, column in enumerate(columns):
            sns.histplot(df[column], kde=True, color='skyblue', bins=10, ax=axes[i])
            if selected_name in df['Common Name'].values:
                selected_value = df[df['Common Name'] == selected_name][column].iloc[0]
                axes[i].axvline(x=selected_value, color='red', linestyle='--', linewidth=2)
            
            axes[i].set_title(f"Distribution of {column}")
            axes[i].set_xlabel(column)
            axes[i].set_ylabel("Frequency")
        
        plt.tight_layout()
        return fig  # Return the figure object for further use 

def carving_evaluation(carving_image_url,carving_source, years_experience, skill_level):

    if not api_key:
        st.error("OpenAI API key is not set. Please set it in your environment variables.")
        return

    # Prompt and Returns

    sysprompt = f"""Please act as the expert woodcarving mentor, Woodshu. You are a master of small figurine, chip, and relief carving, aiming to widely promote the
                    craft through patient and friendly constructive criticism. You enjoy tree-related puns but prioritize insightful critique. 
                    It is understood that users know you are an AI. 
                    Please evaluate the submitted woodcarving by assessing space usage, texture execution, form development, value application, shape definition, and line quality. 
                    Consider each element's contribution to the overall composition, depth, visual appeal, tactile quality, and narrative depth. 
                    Provide an overall score out of 10, adjusted for the carver's reported skill level. 
                    Offer concise, actionable advice on the top three improvement areas, prioritizing suggestions that enhance both the carving's quality and the carver's skills. 
                    Responses should be insightful, friendly, and supportive, fostering constructive criticism and encouraging continual learning and growth.
                    Responses should use words like you and yours if the carver is me, but use they, theirs, etc, if the carver is another. 
                    """
    userprompt = f""" I want to better understand the nuances of this piece and how I can apply it to my own craft. To help you, I should also inform you about the following:
                      The Carver: {carving_source}
                      The Carver's Estimated Years of Experience: {years_experience} 
                      The Carver's Estimated Skill Level: {skill_level}
                    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role":"system",
                "content":sysprompt
                
            },
            {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url": carving_image_url,
                    "detail": "high"
                },
                },
            ],
            },
            {
                "role": "user",
                "content": userprompt
            }
        ],
        max_tokens=2000,
        )
    
    return response.choices[0].message.content

def project_generation(projectType, projectGoal, mediumOfInterst, skillLevel, yearsOfExperience, availableTime):

    if not api_key:
        st.error("OpenAI API key is not set. Please set it in your environment variables.")
        return
    
    model = "gpt-4-turbo"

    sysprompt = f"""Please act as the expert woodworking mentor, Woodshu. You are a master of small figurine, chip, and relief carving, woodburning, and general woodworking.
                    Your goals include educating and widely promoting wood craft through patient and friendly delivery of wood-related information. 
                    You enjoy tree-related puns but prioritize insightful responses.
                """
    
    userprompt = f""" Please help me find my next project by generating a woodworking project idea based on the following criteria:
                 - Project Type: {projectType}
                 - Project Goal: {projectGoal}
                 - Medium of Interest {mediumOfInterst}
                 - My Skill Level: {skillLevel}
                 - Years of Experience {yearsOfExperience}
                 - Available Time (in hours) {availableTime}
                 Please provide a detailed description of the project, including techniques and tools that might be needed. Further, outline the key stages of the project to help track progress and make adjustments as needed.
                 """
    messages = [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": userprompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2  # Lower temperature for less random responses
    )
    return response.choices[0].message.content

def wood_history(wood):
    if not api_key:
        st.error("OpenAI API key is not set. Please set it in your environment variables.")
        return
    
    model = "gpt-4-turbo"

    sysprompt = f"""Please act as the expert woodcarving mentor, Woodshu. You are a master of small figurine, chip, and relief carving, aiming to widely promote the
                    craft through patient and friendly delivery of wood-related information. You enjoy tree-related puns but prioritize insightful responses.
                    """
    userprompt  = f""" I want to better understand about the following type of wood: {wood}. Please help me by formatting your response as indicated below, delimited by ```. Further, please interpret the text delimted between [] as additional instructions, but please remove the [] in the final formatting.
                    ```
                    Wood being Summarized: [{wood}]
                    Wood Type: [Soft or Hardwood]
                    Carvability: [Low, Medium, High, Not Suited for Carving]
                    Recommended Carving Level: [Beginner, Intermediate, Advanced, Not Suited for Carving]
                    Key Woodworking Uses: [Flooring, Decoration, Construction, or other similar keywords]
                    Possible Safety Concerns: [For example, if the sawdust is particularly harmful, the sap caustic, etc. these should be outlined here, otherwise respond with "None"]
                    History and Summary: 
                    [This section should briefly detail the history of the wood's discovery, some botantical facts such as height and life expectancy, as well as the region(s) it is most common to.]
                    Next Steps:
                    To learn about a different wood, please select a new wood from the dropdown menu, then select the "Tell me about..." button. 
                    ```
                    """
    messages = [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": userprompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.15  # Lower temperature for less random responses
    )
    return response.choices[0].message.content

# Initialize pages
def main():
    st.sidebar.title("Woodshu: Your Woodcarving Tutor")
    st.sidebar.subheader("Select a function:")

    # Different functionalities of Woodshu as separate pages - refactor learner vs masterwork to be a "carving critique page that has a selector for your work or other's work"
    pages = {
        "Carving Critique": carving_critique,
        "Project Creation": project_creation,
        "Wood Wonderland": wood_wonderland
    }

    # Dropdown to select the function
    page = st.sidebar.selectbox("Choose the function", list(pages.keys()))

    # Call the appropriate function based on user selection
    pages[page]()

def carving_critique():
    st.title("Carving Critique")

    # Initialize session state keys if they don't exist
    if 'carving_eval_button_pressed' not in st.session_state:
        st.session_state['carving_eval_button_pressed'] = False

    if 'evaluation_result' not in st.session_state:
        # User inputs
        user_image = st.text_input("Enter a carving image URL here please", "https://tinyurl.com/3kwxcfwp")
        reported_years_experience = st.sidebar.slider("Estimated Carver Years of Experience", 0, 20, 10)
        reported_skill_level = st.sidebar.selectbox("Estimated Carver Skill Level", ["Beginner", "Intermediate", "Advanced"], index=2)
        reported_carving_source = st.sidebar.selectbox("Carving Belongs To", ["Me", "Another"], index=1)

        # Button to evaluate carving
        if st.button("Evaluate My Carving") and not st.session_state['carving_eval_button_pressed']:
            st.session_state['carving_eval_button_pressed'] = True  # Button has been pressed
            if user_image:
                st.session_state['uploaded_image'] = user_image


            # Simulate a response from an evaluation function
            with st.spinner('Getting your feedback...'):
                response = carving_evaluation(user_image,reported_carving_source,reported_years_experience,reported_skill_level)
                st.session_state['evaluation_result'] = response

    if 'evaluation_result' in st.session_state:
        st.image(st.session_state['uploaded_image'], caption='Your Submission')
        with st.expander("View Your Evaluation Results"):
            st.text_area("The evaluation", st.session_state['evaluation_result'])
            if st.button("Clear Results", key='clear_results'):
                # Clear specific session state entries
                del st.session_state['uploaded_image']
                del st.session_state['evaluation_result']
                del st.session_state['button_pressed']
                st.rerun()

def project_creation():
    st.title("Project Creation")
    project_type = st.sidebar.selectbox("Project Type", ["Woodcarving", "Woodburning", "Woodworking"], index = 0)
    project_goal = st.sidebar.selectbox("Project Goal", ["Craft Enjoyment", "Skillbuilding", "Show Piece"], index = 0)
    medium_of_interest = st.sidebar.text_input("What kind of thing would you like to work on (e.g. Box, Dragon, etc.)?", "A Watch Box")
    your_skill_level = st.sidebar.selectbox("Your Skill", ["Beginner", "Intermediate", "Advanced"], index = 0)
    your_years_experience = st.sidebar.slider("Years Experience", 0, 50, 1)
    estimated_time = st.sidebar.slider("Time Available (hours)", 1, 100, 10)

    if 'project_button_pressed' not in st.session_state:
        st.session_state['project_button_pressed'] = False

    if 'project_result' not in st.session_state:

        # Button to generate project idea
        if st.button("Generate a Project for Me") and not st.session_state['project_button_pressed']:
            st.session_state['project_button_pressed'] = True
            with st.spinner('Developing your project...'):
                response = project_generation(project_type,project_goal,medium_of_interest,your_skill_level,your_years_experience,estimated_time)
                st.session_state['project_result'] = response
    
    if 'project_result' in st.session_state:
        
        st.write(st.session_state['project_result'])
        
        if st.button("Clear Project", key='clear_project'):
                # Clear specific session state entries
                del st.session_state['project_button_pressed']
                del st.session_state['project_result']
                st.rerun()
    

def wood_wonderland():
    
    # Function to plot distribution for each index
    def plot_distributions(df, selected_name, columns):
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # Adjust layout size based on your need
        axes = axes.flatten()
        
        for i, column in enumerate(columns):
            sns.histplot(df[column], kde=True, color='skyblue', bins=10, ax=axes[i])
            if selected_name in df['Common Name'].values:
                selected_value = df[df['Common Name'] == selected_name][column].iloc[0]
                axes[i].axvline(x=selected_value, color='red', linestyle='--', linewidth=2)
            
            axes[i].set_title(f"Distribution of {column}")
            axes[i].set_xlabel(column)
            axes[i].set_ylabel("Frequency")
        
        plt.tight_layout()
        return fig  # Return the figure object for further use
    
    def writeLoop(woodTable, woodPlot, woodNarrative):
            st.write(woodTable)
            st.pyplot(woodPlot)
            st.write(woodNarrative)
            return True
    
    # Load CSV file
    csv_filename = "woodHardness.csv"
    df = load_data(csv_filename)

    woodData = df[['Common Name', 'Strength Index', 'Janka Index', 'MOE Index', 'MOR Index', 'Crush Index']]

    # Get unique values from a column
    column_name = "Common Name"
    options = woodData[column_name].unique()

    # Dropdown selector
    st.write("All data credit belongs to wood-database.com. Please go support him - the resource is absolutely amazing!")
    selected_option = st.selectbox("Select an option:", options, key = 'wood_type')

    # Initialization of state variables

    if 'wood_details_button_pressed' not in st.session_state:
        st.session_state['wood_details_button_pressed'] = False

    # Button to display wood details
    if not st.session_state['wood_details_button_pressed']:
        if st.button("Tell me more about this wood species!", key='get_wood_details'):
            st.session_state['wood_details_button_pressed'] = True  # Button has been pressed
            st.session_state['wood_numeric_summary'] = woodData[woodData[column_name] == selected_option]

        if st.session_state['wood_details_button_pressed']:
            st.write("Here you are, you eager beaver!")
            # Simulate a response from an evaluation function
            with st.spinner('Summarizing key wood details...'):
                woodInfoGraph = plot_distributions(woodData, selected_option, woodData.columns[1:])
                response = wood_history(selected_option)
                # Uncomment the next line in production
                # response = carving_evaluation(user_image, carving_source, years_experience, skill_level)
                st.session_state['wood_info_results'] = response
                st.session_state['last_option'] = selected_option
                st.rerun()

    # Displaying results if they exist in the session state
    if 'wood_info_results' in st.session_state:
        writeLoop(st.session_state['wood_numeric_summary'], plot_distributions(woodData, st.session_state['last_option'], woodData.columns[1:]), st.session_state['wood_info_results'])

        if st.button("Tell me more about my new wood species!", key='get_wood_details'):
            with st.spinner('Summarizing key wood details...'):
                woodTable = woodData[woodData[column_name] == selected_option]
                st.session_state['wood_numeric_summary'] = woodTable
                st.session_state['wood_info_results'] = wood_history(selected_option)
                st.session_state['last_option'] = selected_option
                st.rerun()

        if st.button("Clear Details", key='clear_wood_details'):
            # Clear session state and reset the app
            for key in ['wood_info_results', 'wood_numeric_summary', 'wood_details_button_pressed']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Run the app
if __name__ == "__main__":
    main()
