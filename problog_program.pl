0::professional_athlete.
0::childhood_entrepreneurship.
0::competitions.
0::ten_thousand_hours_of_mastery.
1::languages.
1::perseverance.
1::risk_tolerance.
1::vision.
1::adaptability.
0::personal_branding.
0::education_level.
0::education_institution.
0::education_field_of_study.
0::education_international_experience.
0::education_extracurricular_involvement.
0::education_awards_and_honors.
0::big_leadership.
0::nasdaq_leadership.
0::number_of_leadership_roles.
0::being_lead_of_nonprofits.
1::number_of_roles.
1::number_of_companies.
0::industry_achievements.
0::big_company_experience.
0::nasdaq_company_experience.
0::big_tech_experience.
0::google_experience.
0::facebook_meta_experience.
0::microsoft_experience.
0::amazon_experience.
0::apple_experience.
1::career_growth.
0::moving_around.
0::international_work_experience.
0::worked_at_military.
0::big_tech_position.
0::worked_at_consultancy.
0::worked_at_bank.
0::press_media_coverage_count.
0::vc_experience.
0::angel_experience.
0::quant_experience.
0::board_advisor_roles.
0::tier_1_vc_experience.
1::startup_experience.
0::ceo_experience.
0::investor_quality_prior_startup.
1::previous_startup_funding_experience.
0::ipo_experience.
0::num_acquisitions.
1::domain_expertise.
1::skill_relevance.
1::yoe.
0.39::success :- previous_startup_funding_experience,investor_quality_prior_startup.
0.29::success :- press_media_coverage_count,personal_branding.
0.29::success :- number_of_leadership_roles,board_advisor_roles.
0.18::success :- big_tech_experience,career_growth.
0.19::success :- vision,big_company_experience.
0.16::success :- international_work_experience,education_level.
0.16::success :- big_tech_position,google_experience.
0.17::success :- education_field_of_study,education_international_experience.
0.21::success :- vc_experience,angel_experience.
0.23::success :- num_acquisitions,ipo_experience.
0.07::success :- industry_achievements,yoe.
0.98::failure :- \+previous_startup_funding_experience,\+investor_quality_prior_startup.
0.93::failure :- \+press_media_coverage_count,\+personal_branding.
0.96::failure :- \+number_of_leadership_roles,\+board_advisor_roles.
0.91::failure :- \+big_tech_experience,\+career_growth.
0.94::failure :- \+vision,\+big_company_experience.
0.96::failure :- \+international_work_experience,\+education_level.
0.95::failure :- \+big_tech_position,\+google_experience.
0.95::failure :- \+education_field_of_study,\+education_international_experience.
0.97::failure :- \+vc_experience,\+angel_experience.
0.96::failure :- \+num_acquisitions,\+ipo_experience.
0.97::failure :- \+industry_achievements,\+yoe.

query(success).

query(failure).
