"""
from pyDatalog import pyDatalog

pyDatalog.create_terms('Founder, Success, strong_technical_background, consumer_platform_experience, entrepreneurial_experience, good_market_timing, leadership_growth')
pyDatalog.create_terms('has_lead_role, years_experience, worked_on_platform, relevant_to_industry, previous_startup, years_as_founder, startup_year, career_progression','Y')

# Define the rules
Success(Founder) <= (
    strong_technical_background(Founder) &
    consumer_platform_experience(Founder) &
    entrepreneurial_experience(Founder) &
    good_market_timing(Founder) &
    leadership_growth(Founder)
)

# Rule 1: Strong technical background
strong_technical_background(Founder) <= (has_lead_role(Founder) & years_experience(Founder, Y) & (Y >= 5))

# Rule 2: Consumer-facing platform experience
consumer_platform_experience(Founder) <= (worked_on_platform(Founder, 'consumer') & relevant_to_industry(Founder, 'mobile_gaming'))

# Rule 3: Entrepreneurial experience
entrepreneurial_experience(Founder) <= (previous_startup(Founder, Y) & years_as_founder(Founder, Y) & (Y >= 3))

# Rule 4: Good market timing (e.g., early-stage entry into high-growth markets)
good_market_timing(Founder) <= (startup_year(Founder, Y) & (2008 <= Y) & (Y <= 2012))

# Rule 5: Leadership growth trajectory
leadership_growth(Founder) <= career_progression(Founder, 'software_developer', 'lead_developer')

# Facts for Ankur Bulsara
+ has_lead_role('ankur_bulsara')
+ years_experience('ankur_bulsara', 7)
+ worked_on_platform('ankur_bulsara', 'consumer')
+ relevant_to_industry('ankur_bulsara', 'mobile_gaming')
+ previous_startup('ankur_bulsara', 'brainwave_software')
+ years_as_founder('ankur_bulsara', 5)
+ startup_year('ankur_bulsara', 2010)
+ career_progression('ankur_bulsara', 'software_developer', 'lead_developer')

# Check if the company is predicted to be successful
print(Success('ankur_bulsara'))
"""