import pandas as pd
import os
from typing import Dict, List, Tuple

class ONETDataLoader:
    def __init__(self, data_dir: str = "db_29_3_excel"):
        self.data_dir = data_dir
        self.occupations_df = None
        self.skills_df = None
        self.interests_df = None
        self.abilities_df = None
        self.knowledge_df = None
        self.work_activities_df = None
        self.work_styles_df = None
        self.work_values_df = None
        self.load_data()
    
    def load_data(self):
        """Load all relevant O*NET data files."""
        try:
            # Load core occupation data
            self.occupations_df = pd.read_excel(
                os.path.join(self.data_dir, "Occupation Data.xlsx")
            )
            
            # Load skills data
            self.skills_df = pd.read_excel(
                os.path.join(self.data_dir, "Skills.xlsx")
            )
            
            # Load interests data
            self.interests_df = pd.read_excel(
                os.path.join(self.data_dir, "Interests.xlsx")
            )
            
            # Load abilities data
            self.abilities_df = pd.read_excel(
                os.path.join(self.data_dir, "Abilities.xlsx")
            )
            
            # Load knowledge data
            self.knowledge_df = pd.read_excel(
                os.path.join(self.data_dir, "Knowledge.xlsx")
            )
            
            # Load work activities
            self.work_activities_df = pd.read_excel(
                os.path.join(self.data_dir, "Work Activities.xlsx")
            )
            
            # Load work styles
            self.work_styles_df = pd.read_excel(
                os.path.join(self.data_dir, "Work Styles.xlsx")
            )
            
            # Load work values
            self.work_values_df = pd.read_excel(
                os.path.join(self.data_dir, "Work Values.xlsx")
            )
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def get_occupation_details(self, occupation_code: str) -> Dict:
        """
        Get comprehensive details for a specific occupation.
        
        Args:
            occupation_code: O*NET occupation code
            
        Returns:
            Dictionary containing occupation details
        """
        occupation = self.occupations_df[
            self.occupations_df['O*NET-SOC Code'] == occupation_code
        ].iloc[0]
        
        # Get skills
        skills = self.skills_df[
            self.skills_df['O*NET-SOC Code'] == occupation_code
        ][['Element Name', 'Scale ID', 'Data Value']].to_dict('records')
        
        # Get interests
        interests = self.interests_df[
            self.interests_df['O*NET-SOC Code'] == occupation_code
        ][['Element Name', 'Scale ID', 'Data Value']].to_dict('records')
        
        # Get abilities
        abilities = self.abilities_df[
            self.abilities_df['O*NET-SOC Code'] == occupation_code
        ][['Element Name', 'Scale ID', 'Data Value']].to_dict('records')
        
        # Get knowledge
        knowledge = self.knowledge_df[
            self.knowledge_df['O*NET-SOC Code'] == occupation_code
        ][['Element Name', 'Scale ID', 'Data Value']].to_dict('records')
        
        return {
            'title': occupation['Title'],
            'description': occupation['Description'],
            'skills': skills,
            'interests': interests,
            'abilities': abilities,
            'knowledge': knowledge
        }
    
    def get_all_occupations(self) -> List[Dict]:
        """
        Get a list of all occupations with basic information.
        
        Returns:
            List of dictionaries containing occupation information
        """
        return self.occupations_df[[
            'O*NET-SOC Code',
            'Title',
            'Description'
        ]].to_dict('records')
    
    def get_occupation_by_skills(self, skills: List[str]) -> List[Tuple[str, float]]:
        """
        Find occupations that match given skills.
        
        Args:
            skills: List of skills to match
            
        Returns:
            List of tuples (occupation_code, match_score)
        """
        # Implementation to be added
        pass
    
    def get_occupation_by_interests(self, interests: List[str]) -> List[Tuple[str, float]]:
        """
        Find occupations that match given interests.
        
        Args:
            interests: List of interests to match
            
        Returns:
            List of tuples (occupation_code, match_score)
        """
        # Implementation to be added
        pass 