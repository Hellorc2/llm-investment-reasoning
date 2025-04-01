0::professional_athlete.
0::childhood_entrepreneurship.
0::competitions.
0::ten_thousand_hours_of_mastery.
1::languages.
1::perseverance.
0::risk_tolerance.
1::vision.
1::adaptability.
0::personal_branding.
0::education_level.
0::education_institution.
0::education_field_of_study.
0::education_international_experience.
0::education_extracurricular_involvement.
0::education_awards_and_honors.
1::big_leadership.
0::nasdaq_leadership.
1::number_of_leadership_roles.
0::being_lead_of_nonprofits.
0::number_of_roles.
0::number_of_companies.
0::industry_achievements.
0::big_company_experience.
0::nasdaq_company_experience.
0::big_tech_experience.
0::google_experience.
0::facebook_meta_experience.
0::microsoft_experience.
0::amazon_experience.
0::apple_experience.
0::career_growth.
0::moving_around.
0::international_work_experience.
0::worked_at_military.
0::big_tech_position.
0::worked_at_consultancy.
0::worked_at_bank.
1::press_media_coverage_count.
0::vc_experience.
0::angel_experience.
0::quant_experience.
1::board_advisor_roles.
0::tier_1_vc_experience.
0::startup_experience.
0::ceo_experience.
0::investor_quality_prior_startup.
0::previous_startup_funding_experience.
0::ipo_experience.
0::num_acquisitions.
1::domain_expertise.
1::skill_relevance.
0::yoe.
0.31::success :- startup_experience,num_acquisitions.
0.27::success :- previous_startup_funding_experience,investor_quality_prior_startup.
0.15::success :- big_tech_experience,big_tech_position.
0.35::success :- ceo_experience,board_advisor_roles.
0.26::success :- nasdaq_company_experience,nasdaq_leadership.
0.23::success :- big_company_experience,number_of_leadership_roles.
0.18::success :- press_media_coverage_count,personal_branding.
0.10::success :- domain_expertise,skill_relevance.
0.13::success :- education_level,education_institution.
0.12::success :- education_field_of_study,education_international_experience.
0.20::success :- vc_experience,tier_1_vc_experience.
0.96::failure :- \+education_level,\+education_institution.
0.91::failure :- \+startup_experience,\+ceo_experience.
0.91::failure :- \+big_company_experience,\+nasdaq_company_experience.
0.91::failure :- number_of_roles,moving_around.
0.92::failure :- \+press_media_coverage_count,\+personal_branding.
0.97::failure :- \+domain_expertise,\+skill_relevance.
0.91::failure :- \+big_tech_experience,\+big_tech_position.
0.96::failure :- \+education_field_of_study,\+international_work_experience.
0.91::failure :- \+previous_startup_funding_experience,\+investor_quality_prior_startup.
0.91::failure :- \+num_acquisitions,\+startup_experience.
0.93::failure :- \+yoe,\+industry_achievements.

query(success).

query(failure).
